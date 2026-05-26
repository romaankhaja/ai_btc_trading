"""
Paper Trader — Event-driven simulation loop.

Processes historical candles one-by-one through the full
inference pipeline, simulating realistic execution with
slippage, fees, and order tracking.
"""

import logging
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

from inference.trade_decision import TradeDecisionEngine, TradeDecision
from training.config import (
    MIN_CONFIDENCE,
    MAX_HOLDING_BARS,
    REGIME_MAX_BARS,
    NO_TRADE_REGIMES,
)

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """A single simulated trade."""
    entry_time: pd.Timestamp
    entry_price: float
    direction: int            # +1 LONG, -1 SHORT
    sl_price: float
    tp_price: float
    risk_percent: float
    position_size_usd: float
    regime: str
    confidence: float
    atr_at_entry: float = 0.0
    atr_ratio_at_entry: float = 0.0
    raw_regime: str = ''
    # Filled on exit
    exit_time: Optional[pd.Timestamp] = None
    exit_price: float = 0.0
    pnl: float = 0.0
    exit_reason: str = ''


@dataclass
class PaperTradingResult:
    """Full paper trading simulation results."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    avg_rr_realized: float = 0.0
    trade_frequency_pct: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    regime_performance: dict = field(default_factory=dict)
    block_reasons_summary: dict = field(default_factory=dict)
    circuit_breaker_activations: int = 0
    regime_override_count: int = 0


class PaperTrader:
    """
    Event-driven paper trading simulator.
    
    Processes each candle through the full inference pipeline:
    features -> models -> policy -> threshold -> sizing -> execution
    """
    
    FEE_PCT = 0.0005        # 0.05% taker fee per trade (Standard Futures Taker Fee)
    SL_COOLDOWN_BARS = 12
    
    def __init__(self, models_dir: str, initial_equity: float = 10000.0):
        self.engine = TradeDecisionEngine(models_dir)
        self.initial_equity = initial_equity
    
    def load(self):
        """Load inference models."""
        self.engine.load()
    
    def run(self, df: pd.DataFrame) -> PaperTradingResult:
        """
        Run the full paper trading simulation.
        
        Args:
            df: DataFrame with all features (test set)
        
        Returns:
            PaperTradingResult with comprehensive metrics
        """
        result = PaperTradingResult()
        equity = self.initial_equity
        peak_equity = equity
        open_trade: Optional[Trade] = None
        all_returns = []
        block_counter = {}
        skipped_trade_events = []
        regime_override_events = []
        atr_history = deque(maxlen=20)
        bars_held = 0
        consecutive_sl_count = 0
        sl_cooldown_remaining_bars = 0
        drawdown_cooldown_bars = 0
        circuit_breaker_events = []
        last_circuit_state = 'normal'
        last_drawdown_state = 'normal'
        
        columns = list(df.columns)

        def _log_circuit_event(time_value, event_type, detail, streak, cooldown, would_have_overridden=None):
            event = {
                'timestamp': time_value,
                'event_type': event_type,
                'detail': detail,
                'consecutive_sl_count': streak,
                'cooldown_remaining_bars': cooldown,
            }
            if would_have_overridden is not None:
                event['would_have_overridden'] = bool(would_have_overridden)
            circuit_breaker_events.append(event)

        def _update_circuit_state(exit_reason, pnl, time_value, trade: Optional[Trade] = None):
            nonlocal consecutive_sl_count, sl_cooldown_remaining_bars, last_circuit_state, result

            if exit_reason == 'SL_HIT':
                consecutive_sl_count += 1
                if consecutive_sl_count >= 2:
                    sl_cooldown_remaining_bars = self.SL_COOLDOWN_BARS
            elif exit_reason == 'TP_HIT' or (exit_reason == 'TIME_BARRIER' and pnl > 0):
                consecutive_sl_count = 0
                sl_cooldown_remaining_bars = 0

            if sl_cooldown_remaining_bars > 0:
                current_state = 'cooldown'
            elif consecutive_sl_count >= 4:
                current_state = 'full_block'
            elif consecutive_sl_count == 3:
                current_state = 'cooldown_start'
            elif consecutive_sl_count == 2:
                current_state = 'reduce_50'
            else:
                current_state = 'normal'

            if current_state != last_circuit_state:
                if current_state != 'normal':
                    _log_circuit_event(
                        time_value,
                        current_state,
                        f'exit_reason={exit_reason}, pnl={pnl:.2f}',
                        consecutive_sl_count,
                        sl_cooldown_remaining_bars,
                    )
                    result.circuit_breaker_activations += 1
                last_circuit_state = current_state

            if trade is not None and trade.regime in ('trending_up', 'trending_down') and exit_reason == 'SL_HIT':
                would_have_overridden = bool(trade.atr_ratio_at_entry > 1.4)
                _log_circuit_event(
                    time_value,
                    'sl_retro',
                    f'would_have_overridden={would_have_overridden}, atr_ratio={trade.atr_ratio_at_entry:.2f}',
                    consecutive_sl_count,
                    sl_cooldown_remaining_bars,
                    would_have_overridden=would_have_overridden,
                )

        for i in range(len(df)):
            row = df.iloc[i]
            features = {col: float(row[col]) if isinstance(row[col], (int, float, np.integer, np.floating)) else row[col] for col in columns}
            current_time = row.get('open_time', pd.Timestamp.now())
            current_close = row['close']
            atr = float(row.get('atr_14', 0.0))
            atr_history.append(atr)
            atr_20_mean = float(np.mean(atr_history)) if len(atr_history) > 0 else 0.0
            features['consecutive_sl_count'] = consecutive_sl_count
            features['sl_cooldown_bars_remaining'] = sl_cooldown_remaining_bars
            features['atr_20bar_mean'] = atr_20_mean
            
            # ---- Check open trade for SL/TP hit ----
            if open_trade is not None:
                # 0. Trailing Stop-Loss Logic (Fix 10)
                # When price moves in favor of the trade by more than 1.5 ATR,
                # set the stop-loss to entry + 0.5 ATR (for Long) or entry - 0.5 ATR (for Short)
                atr_ent = open_trade.atr_at_entry
                if open_trade.direction == 1:  # LONG
                    if row['high'] >= open_trade.entry_price + 1.5 * atr_ent:
                        new_sl = open_trade.entry_price + 0.5 * atr_ent
                        if new_sl > open_trade.sl_price:
                            open_trade.sl_price = new_sl
                else:  # SHORT
                    if row['low'] <= open_trade.entry_price - 1.5 * atr_ent:
                        new_sl = open_trade.entry_price - 0.5 * atr_ent
                        if new_sl < open_trade.sl_price:
                            open_trade.sl_price = new_sl
                
                hit = False
                target_price = 0.0
                
                if open_trade.direction == 1:  # LONG
                    if row['low'] <= open_trade.sl_price:
                        target_price = open_trade.sl_price
                        open_trade.exit_reason = 'SL_HIT'
                        hit = True
                    elif row['high'] >= open_trade.tp_price:
                        target_price = open_trade.tp_price
                        open_trade.exit_reason = 'TP_HIT'
                        hit = True
                else:  # SHORT
                    if row['high'] >= open_trade.sl_price:
                        target_price = open_trade.sl_price
                        open_trade.exit_reason = 'SL_HIT'
                        hit = True
                    elif row['low'] <= open_trade.tp_price:
                        target_price = open_trade.tp_price
                        open_trade.exit_reason = 'TP_HIT'
                        hit = True
                
                if hit:
                    # Apply dynamic slippage (Fix 10: 5% of ATR at exit)
                    exit_slip = 0.05 * atr
                    open_trade.exit_price = target_price - open_trade.direction * exit_slip
                    
                    # Apply transaction fees on entry and exitlegs
                    entry_fee = open_trade.entry_price * self.FEE_PCT
                    exit_fee = open_trade.exit_price * self.FEE_PCT
                    
                    # PnL Calculation
                    raw_pnl = open_trade.direction * (open_trade.exit_price - open_trade.entry_price)
                    pnl_per_unit = raw_pnl - entry_fee - exit_fee
                    
                    # Scale to position size
                    units = open_trade.position_size_usd / open_trade.entry_price if open_trade.entry_price > 0 else 0
                    open_trade.pnl = pnl_per_unit * units
                    open_trade.exit_time = current_time
                    
                    equity += open_trade.pnl
                    all_returns.append(open_trade.pnl / peak_equity if peak_equity > 0 else 0)
                    
                    result.trades.append(open_trade)
                    _update_circuit_state(open_trade.exit_reason, open_trade.pnl, current_time, open_trade)
                    features['consecutive_sl_count'] = consecutive_sl_count
                    features['sl_cooldown_bars_remaining'] = sl_cooldown_remaining_bars
                    open_trade = None
                    bars_held = 0
                else:
                    bars_held += 1
                    # Mid-trade momentum reversal check every 4 bars.
                    if bars_held % 4 == 0:
                        rsi_velocity = float(row.get('rsi_velocity', 0.0))
                        near_entry = abs(current_close - open_trade.entry_price) / open_trade.entry_price <= 0.001 if open_trade.entry_price > 0 else False
                        rsi_flip = (
                            (open_trade.direction == 1 and rsi_velocity < 0)
                            or (open_trade.direction == -1 and rsi_velocity > 0)
                        )
                        if rsi_flip and near_entry:
                            open_trade.exit_reason = 'MOMENTUM_REVERSAL'
                            exit_slip = 0.05 * atr
                            open_trade.exit_price = current_close - open_trade.direction * exit_slip
                            entry_fee = open_trade.entry_price * self.FEE_PCT
                            exit_fee = open_trade.exit_price * self.FEE_PCT
                            raw_pnl = open_trade.direction * (open_trade.exit_price - open_trade.entry_price)
                            pnl_per_unit = raw_pnl - entry_fee - exit_fee
                            units = open_trade.position_size_usd / open_trade.entry_price if open_trade.entry_price > 0 else 0
                            open_trade.pnl = pnl_per_unit * units
                            open_trade.exit_time = current_time
                            equity += open_trade.pnl
                            all_returns.append(open_trade.pnl / peak_equity if peak_equity > 0 else 0)
                            result.trades.append(open_trade)
                            _update_circuit_state(open_trade.exit_reason, open_trade.pnl, current_time, open_trade)
                            features['consecutive_sl_count'] = consecutive_sl_count
                            features['sl_cooldown_bars_remaining'] = sl_cooldown_remaining_bars
                            open_trade = None
                            bars_held = 0
                        else:
                            # Time barrier by regime-specific maximum bars.
                            current_max_bars = REGIME_MAX_BARS.get(open_trade.regime, MAX_HOLDING_BARS)
                            if bars_held >= current_max_bars:
                                open_trade.exit_reason = 'TIME_BARRIER'
                                
                                # Apply slippage on exit close price (5% of ATR)
                                exit_slip = 0.05 * atr
                                open_trade.exit_price = current_close - open_trade.direction * exit_slip
                                
                                entry_fee = open_trade.entry_price * self.FEE_PCT
                                exit_fee = open_trade.exit_price * self.FEE_PCT
                                
                                raw_pnl = open_trade.direction * (open_trade.exit_price - open_trade.entry_price)
                                pnl_per_unit = raw_pnl - entry_fee - exit_fee
                                
                                units = open_trade.position_size_usd / open_trade.entry_price if open_trade.entry_price > 0 else 0
                                open_trade.pnl = pnl_per_unit * units
                                open_trade.exit_time = current_time
                                
                                equity += open_trade.pnl
                                all_returns.append(open_trade.pnl / peak_equity if peak_equity > 0 else 0)
                                
                                result.trades.append(open_trade)
                                _update_circuit_state(open_trade.exit_reason, open_trade.pnl, current_time, open_trade)
                                features['consecutive_sl_count'] = consecutive_sl_count
                                features['sl_cooldown_bars_remaining'] = sl_cooldown_remaining_bars
                                open_trade = None
                                bars_held = 0
            
            # ---- Make new decision if no open trade ----
            if open_trade is None:
                if drawdown_cooldown_bars > 0:
                    all_returns.append(0.0)
                    block_counter['DRAWDOWN_BLOCK'] = block_counter.get('DRAWDOWN_BLOCK', 0) + 1
                    drawdown_cooldown_bars -= 1
                else:
                    decision = self.engine.decide(features, equity)

                    if decision.regime_override_applied:
                        regime_override_events.append({
                            'timestamp': current_time,
                            'raw_regime': decision.raw_regime,
                            'overridden_regime': decision.regime,
                            'atr_ratio': decision.atr_ratio,
                            'reason': decision.regime_override_reason,
                        })
                        result.regime_override_count += 1

                    allow_entry = decision.action in ('LONG', 'SHORT')
                    skip_reason = ''
                    if allow_entry:
                        current_regime = decision.regime
                        breakout_allow = (
                            current_regime == 'ranging'
                            and float(row.get('bb_width_percentile', 0.0)) > 0.70
                            and float(row.get('volume_spike_score', 0.0)) > 1.5
                        )

                        if current_regime in NO_TRADE_REGIMES and not breakout_allow:
                            allow_entry = False
                            skip_reason = f'NO_TRADE_REGIME: {current_regime}'

                        if allow_entry and decision.meta_probability < MIN_CONFIDENCE:
                            allow_entry = False
                            skip_reason = (
                                f'CONFIDENCE_GATE: confidence={decision.meta_probability:.3f} < {MIN_CONFIDENCE:.2f}'
                            )

                        if allow_entry:
                            expected_rr = float(decision.reward_risk_ratio)
                            if expected_rr < 1.3:
                                allow_entry = False
                                skip_reason = f'RR_GATE: expected_rr={expected_rr:.2f} < 1.30'

                        if allow_entry and current_regime in ('trending_up', 'trending_down'):
                            ema_20_slope = float(row.get('ema_20_slope', 0.0))
                            trend_alignment = float(row.get('trend_alignment_score', 0.0))
                            trade_dir = 1 if decision.action == 'LONG' else -1
                            if trend_alignment <= 0.6:
                                allow_entry = False
                                skip_reason = (
                                    f'TREND_GATE: trend_alignment_score={trend_alignment:.2f} <= 0.60'
                                )
                            elif trade_dir == 1 and ema_20_slope <= 0:
                                allow_entry = False
                                skip_reason = (
                                    f'TREND_GATE: ema_20_slope={ema_20_slope:.4f} not supportive for LONG'
                                )
                            elif trade_dir == -1 and ema_20_slope >= 0:
                                allow_entry = False
                                skip_reason = (
                                    f'TREND_GATE: ema_20_slope={ema_20_slope:.4f} not supportive for SHORT'
                                )

                    if allow_entry:
                        direction = 1 if decision.action == 'LONG' else -1

                        # Dynamic entry slippage: 5% of ATR (Fix 10)
                        entry_slip = 0.05 * atr
                        entry_with_slip = current_close + direction * entry_slip

                        open_trade = Trade(
                            entry_time=current_time,
                            entry_price=entry_with_slip,
                            direction=direction,
                            sl_price=decision.sl_price,
                            tp_price=decision.tp_price,
                            risk_percent=decision.risk_percent,
                            position_size_usd=decision.position_size_usd,
                            regime=decision.regime,
                            confidence=decision.meta_probability,
                            atr_at_entry=atr,
                            atr_ratio_at_entry=decision.atr_ratio,
                            raw_regime=decision.raw_regime,
                        )
                        bars_held = 0
                    else:
                        all_returns.append(0.0)
                        if decision.action in ('LONG', 'SHORT'):
                            skipped_trade_events.append({
                                'timestamp': current_time,
                                'action': decision.action,
                                'regime': decision.regime,
                                'confidence': decision.meta_probability,
                                'meta_probability': decision.meta_probability,
                                'zscore': 0.0,
                                'expected_rr': float(decision.reward_risk_ratio),
                                'atr_ratio': decision.atr_ratio,
                                'reason': skip_reason or 'UNKNOWN',
                            })
                            tag = skip_reason.split(':')[0].strip() if skip_reason else 'SKIP'
                            block_counter[tag] = block_counter.get(tag, 0) + 1
                            if skip_reason:
                                decision.block_reasons.append(skip_reason)
                        else:
                            for reason in decision.block_reasons:
                                tag = reason.split(':')[0].strip()
                                block_counter[tag] = block_counter.get(tag, 0) + 1

            session_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            if session_drawdown > 0.04 and drawdown_cooldown_bars == 0:
                drawdown_cooldown_bars = 48
                _log_circuit_event(
                    current_time,
                    'drawdown_block',
                    f'reason=DRAWDOWN_BLOCK, session_drawdown={session_drawdown:.2%}',
                    consecutive_sl_count,
                    sl_cooldown_remaining_bars,
                )
                result.circuit_breaker_activations += 1
                # Reset peak equity to avoid immediate re-triggering once cooldown ends
                peak_equity = equity
            
            # Track equity
            if equity > peak_equity:
                peak_equity = equity
            result.equity_curve.append(equity)

            if sl_cooldown_remaining_bars > 0:
                sl_cooldown_remaining_bars -= 1
        
        # ---- Close any remaining open trade at last close ----
        if open_trade is not None:
            open_trade.exit_price = df.iloc[-1]['close']
            open_trade.exit_reason = 'END_OF_DATA'
            raw_pnl = open_trade.direction * (open_trade.exit_price - open_trade.entry_price)
            units = open_trade.position_size_usd / open_trade.entry_price if open_trade.entry_price > 0 else 0
            open_trade.pnl = raw_pnl * units
            open_trade.exit_time = df.iloc[-1].get('open_time', pd.Timestamp.now())
            equity += open_trade.pnl
            result.trades.append(open_trade)
            _update_circuit_state(open_trade.exit_reason, open_trade.pnl, open_trade.exit_time, open_trade)
            result.equity_curve.append(equity)
        
        # ---- Compute Final Metrics ----
        result.total_trades = len(result.trades)
        result.total_return = (equity - self.initial_equity) / self.initial_equity
        result.trade_frequency_pct = result.total_trades / len(df) * 100 if len(df) > 0 else 0
        
        if result.total_trades > 0:
            wins = sum(1 for t in result.trades if t.pnl > 0)
            result.win_rate = wins / result.total_trades
            avg_win = np.mean([t.pnl for t in result.trades if t.pnl > 0]) if wins > 0 else 0
            losses = result.total_trades - wins
            avg_loss = abs(np.mean([t.pnl for t in result.trades if t.pnl <= 0])) if losses > 0 else 1
            result.avg_rr_realized = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Sharpe & Sortino (annualized for 15m bars)
        returns = np.array(all_returns)
        if len(returns) > 1 and returns.std() > 0:
            result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(35040)
            downside = returns[returns < 0]
            if len(downside) > 0 and downside.std() > 0:
                result.sortino_ratio = (returns.mean() / downside.std()) * np.sqrt(35040)
        
        # Max Drawdown
        eq = np.array(result.equity_curve)
        if len(eq) > 0:
            peak = np.maximum.accumulate(eq)
            dd = (eq - peak) / peak
            result.max_drawdown = float(dd.min())
        
        # Regime performance
        for t in result.trades:
            if t.regime not in result.regime_performance:
                result.regime_performance[t.regime] = {'trades': 0, 'pnl': 0.0, 'wins': 0}
            result.regime_performance[t.regime]['trades'] += 1
            result.regime_performance[t.regime]['pnl'] += t.pnl
            if t.pnl > 0:
                result.regime_performance[t.regime]['wins'] += 1
        
        result.block_reasons_summary = block_counter
        result.circuit_breaker_activations = len(circuit_breaker_events)

        results_dir = Path(__file__).resolve().parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)
        pd.DataFrame(
            circuit_breaker_events,
            columns=['timestamp', 'event_type', 'detail', 'consecutive_sl_count', 'cooldown_remaining_bars', 'would_have_overridden']
        ).to_csv(results_dir / 'circuit_breaker_log.csv', index=False)
        pd.DataFrame(
            skipped_trade_events,
            columns=[
                'timestamp', 'action', 'regime', 'confidence', 'meta_probability',
                'zscore', 'expected_rr', 'atr_ratio', 'reason'
            ]
        ).to_csv(results_dir / 'skipped_trades_log.csv', index=False)
        pd.DataFrame(
            regime_override_events,
            columns=['timestamp', 'raw_regime', 'overridden_regime', 'atr_ratio', 'reason']
        ).to_csv(results_dir / 'regime_override_log.csv', index=False)
        
        return result
