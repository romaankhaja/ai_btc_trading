"""
Non-Homogeneous Hidden Markov Model (NHHMM) for Regime Detection.

Standard Gaussian HMM has fixed transition matrices. The NHHMM extends this
by making the transition matrix a function of exogenous macro covariates:
  - dxy_trend         (US Dollar Index direction)
  - us_10y_roc        (10-Year Treasury yield rate-of-change)
  - btc_spx_corr      (Rolling 30-day BTC-SPX correlation)

Architecture:
  - Emission model  : GaussianHMM (log_returns, realized_vol)
  - Transition model: Logistic regression per (from_state, to_state) pair
    conditioned on the macro covariate vector at time t.
  - Decoding         : Custom Viterbi with time-varying T matrices.

Regime states (3):
  0 → trending
  1 → sideways
  2 → high_risk

Usage:
    from regime.nhhmm import NHHMMRegimeDetector
    detector = NHHMMRegimeDetector(n_states=3)
    detector.fit(train_df)
    train_df['regime_label'] = detector.decode(train_df)
"""

import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Emission features fed into the GaussianHMM
EMISSION_FEATURES = ["log_return", "realized_vol_15m"]

# Transition covariate features
TRANSITION_FEATURES = ["dxy_trend", "us_10y_roc", "btc_spx_corr"]

# Canonical state-name mapping (assigned post-hoc by centroid inspection)
DEFAULT_MAPPING = {
    0: "trending",
    1: "sideways",
    2: "high_risk"
}


class NHHMMRegimeDetector:
    """
    Non-Homogeneous HMM: GaussianHMM emissions + logistic transition covariates.
    """

    def __init__(self, n_states: int = 3, n_iter: int = 100, random_state: int = 42):
        self.n_states = n_states
        self.n_iter   = n_iter
        self.random_state = random_state

        # Fitted objects
        self.hmm_model      = None   # hmmlearn GaussianHMM
        self.emission_scaler = None  # StandardScaler for emission features
        self.trans_scalers   = {}    # {from_state: StandardScaler}
        self.trans_models    = {}    # {from_state: LogisticRegression}
        self.state_mapping   = DEFAULT_MAPPING.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_emissions(self, df: pd.DataFrame) -> np.ndarray:
        """Build emission matrix X from DataFrame."""
        # Compute log_return if absent
        df = df.copy()
        if "log_return" not in df.columns:
            df["log_return"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        if "realized_vol_15m" not in df.columns:
            df["realized_vol_15m"] = df["log_return"].rolling(20).std().fillna(0)
        X = df[EMISSION_FEATURES].values.astype(float)
        # Replace any remaining NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def _prepare_covariates(self, df: pd.DataFrame) -> np.ndarray:
        """Build transition covariate matrix Z from DataFrame."""
        df = df.copy()
        for col in TRANSITION_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
        Z = df[TRANSITION_FEATURES].values.astype(float)
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        return Z

    def _map_states_by_vol(self) -> dict:
        """
        Assign semantic labels to numeric HMM states based on emission means.
        Uses realized_vol_15m (index 1) and log_return (index 0).
        """
        means = self.hmm_model.means_  # (n_states, n_features)
        vols  = means[:, 1]            # realized_vol_15m
        rets  = means[:, 0]            # log_return

        sorted_by_vol = np.argsort(vols)  # ascending volatility
        mapping = {}

        # Lowest-vol states -> sideways and trending (by return)
        low_vol_states = sorted_by_vol[:2]
        rets_low = np.abs(rets[low_vol_states])
        trending = low_vol_states[np.argmax(rets_low)]
        sideways = low_vol_states[np.argmin(rets_low)]

        # highest-vol state -> high_risk
        high_risk = sorted_by_vol[-1]

        mapping[int(trending)]  = "trending"
        mapping[int(sideways)]  = "sideways"
        mapping[int(high_risk)] = "high_risk"

        # Fill any gaps (n_states > 4) with generic label
        for i in range(self.n_states):
            if i not in mapping:
                mapping[i] = f"state_{i}"

        return mapping

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "NHHMMRegimeDetector":
        """
        Fit the NHHMM on the training DataFrame.

        Step 1: Fit Gaussian HMM on emission features.
        Step 2: Decode training sequence → get state sequence.
        Step 3: For each from_state, fit a multinomial logistic regression
                of (Z_t → s_{t+1}) conditioned on transition covariates.
        """
        logger.info(f"Fitting NHHMM (n_states={self.n_states}) on {len(df)} rows...")

        X = self._prepare_emissions(df)
        Z = self._prepare_covariates(df)

        # Scale emissions
        self.emission_scaler = StandardScaler()
        X_scaled = self.emission_scaler.fit_transform(X)

        # Step 1: Fit Gaussian HMM
        self.hmm_model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )
        self.hmm_model.fit(X_scaled)
        logger.info(f"  GaussianHMM converged: {self.hmm_model.monitor_.converged}")

        # Step 2: Decode training sequence
        states = self.hmm_model.predict(X_scaled)

        # Assign semantic labels
        self.state_mapping = self._map_states_by_vol()
        logger.info(f"  State mapping: {self.state_mapping}")

        # Step 3: Fit per-state transition logistic regressors
        T = len(states)
        for s in range(self.n_states):
            # Indices where we ARE in state s (not the last bar)
            from_idx = np.where(states[:-1] == s)[0]
            if len(from_idx) < self.n_states * 2:
                logger.warning(f"  State {s} has too few transitions ({len(from_idx)}); skipping covariate model.")
                continue

            Z_from     = Z[from_idx]         # covariates when in state s
            s_next     = states[from_idx + 1] # next state

            scaler = StandardScaler()
            Z_scaled = scaler.fit_transform(Z_from)

            clf = LogisticRegression(
                solver="lbfgs",
                max_iter=500,
                random_state=self.random_state,
                C=1.0,
            )
            # Need all classes represented; if missing, use fixed prior
            if len(np.unique(s_next)) < 2:
                logger.warning(f"  State {s} only transitions to one state; skipping covariate model.")
                continue

            clf.fit(Z_scaled, s_next)
            self.trans_scalers[s] = scaler
            self.trans_models[s]  = clf

        logger.info(f"  Covariate transition models fitted for states: {list(self.trans_models.keys())}")
        return self

    # ------------------------------------------------------------------
    # Decode (Viterbi with time-varying transition matrices)
    # ------------------------------------------------------------------

    def _build_transition_matrix(self, z: np.ndarray, from_state: int) -> np.ndarray:
        """
        Build the row of the transition matrix for from_state at covariate vector z.
        Returns a probability vector of length n_states.
        """
        if from_state in self.trans_models:
            scaler = self.trans_scalers[from_state]
            clf    = self.trans_models[from_state]
            z_scaled = scaler.transform(z.reshape(1, -1))
            # predict_proba returns probabilities for each class in clf.classes_
            probs_partial = clf.predict_proba(z_scaled)[0]
            full_probs = np.zeros(self.n_states)
            for cls, p in zip(clf.classes_, probs_partial):
                full_probs[int(cls)] = p
            # Smooth with small floor to avoid zero-probability transitions
            full_probs = 0.95 * full_probs + 0.05 * (np.ones(self.n_states) / self.n_states)
            return full_probs / full_probs.sum()
        else:
            # Fall back to the stationary transition row from the HMM
            return self.hmm_model.transmat_[from_state]

    def decode(self, df: pd.DataFrame) -> np.ndarray:
        """
        Custom Viterbi decoding with time-varying transition matrices.
        Returns integer state sequence of length len(df).
        """
        X = self._prepare_emissions(df)
        Z = self._prepare_covariates(df)
        X_scaled = self.emission_scaler.transform(X)
        T = len(X_scaled)

        # Log emission probabilities from the fitted GaussianHMM
        log_emit = self.hmm_model._compute_log_likelihood(X_scaled)  # (T, n_states)

        # Precompute all transition probabilities for all states at all time steps
        trans_probs = np.zeros((T, self.n_states, self.n_states))
        for s in range(self.n_states):
            if s in self.trans_models:
                scaler = self.trans_scalers[s]
                clf = self.trans_models[s]
                Z_scaled = scaler.transform(Z)
                probs_partial = clf.predict_proba(Z_scaled)  # Shape: (T, num_classes)
                full_probs = np.zeros((T, self.n_states))
                for idx, cls in enumerate(clf.classes_):
                    full_probs[:, int(cls)] = probs_partial[:, idx]
                full_probs = 0.95 * full_probs + 0.05 / self.n_states
                row_sums = full_probs.sum(axis=1, keepdims=True)
                full_probs = full_probs / row_sums
                trans_probs[:, s, :] = full_probs
            else:
                trans_probs[:, s, :] = self.hmm_model.transmat_[s]

        # Viterbi DP
        log_delta = np.full((T, self.n_states), -np.inf)
        psi       = np.zeros((T, self.n_states), dtype=int)

        # Initialization
        log_delta[0] = np.log(self.hmm_model.startprob_ + 1e-300) + log_emit[0]

        for t in range(1, T):
            for j in range(self.n_states):
                # trans_probs[t-1] has shape (n_states, n_states). trans_probs[t-1, s, j] is s -> j
                # We want the column of transition probabilities to state j from all s
                trans_col = trans_probs[t-1, :, j]
                scores = log_delta[t-1] + np.log(trans_col + 1e-300) + log_emit[t, j]
                best   = np.argmax(scores)
                log_delta[t, j] = scores[best]
                psi[t, j]       = best

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(log_delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]

        return states

    def assign_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decode regime states and add human-readable label columns.
        Adds: regime_state (int), regime_label (str), regime_confidence (float).
        """
        df = df.copy()
        X = self._prepare_emissions(df)
        X_scaled = self.emission_scaler.transform(X)

        states = self.decode(df)
        df["regime_state"] = states
        df["regime_label"] = [self.state_mapping.get(s, f"state_{s}") for s in states]

        # Confidence = posterior probability of the decoded state (from HMM posteriors)
        log_emit = self.hmm_model._compute_log_likelihood(X_scaled)
        posteriors = np.exp(log_emit - log_emit.max(axis=1, keepdims=True))
        posteriors /= posteriors.sum(axis=1, keepdims=True)
        df["regime_confidence"] = posteriors[np.arange(len(states)), states]

        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_dir / "nhhmm_regime_detector.pkl")
        logger.info(f"Saved NHHMM detector → {output_dir}/nhhmm_regime_detector.pkl")

    @classmethod
    def load(cls, output_dir: Path) -> "NHHMMRegimeDetector":
        path = Path(output_dir) / "nhhmm_regime_detector.pkl"
        obj = joblib.load(path)
        logger.info(f"Loaded NHHMM detector from {path}")
        return obj
