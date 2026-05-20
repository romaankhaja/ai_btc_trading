"""
NHHMM Regime Labeler — Non-Homogeneous Hidden Markov Model.

Wraps the institutional NHHMMRegimeDetector in our training orchestrator,
making the transitions dependent on exogenous covariates.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from regime.nhhmm import NHHMMRegimeDetector

logger = logging.getLogger(__name__)

# Stationary emission features fed into the GaussianHMM
HMM_FEATURES = ["log_return", "realized_vol_15m"]


def prepare_hmm_features(df):
    """
    Ensure all features required for emissions and transitions exist.
    """
    df = df.copy()
    if 'log_return' not in df.columns:
        df['log_return'] = pd.Series(np.log(df['close'] / df['close'].shift(1))).fillna(0.0)
    return df


def fit_hmm_model(train_df, n_components=4, random_state=42):
    """
    Fit Non-Homogeneous HMM on training data ONLY.
    
    Returns:
        detector, detector, detector.state_mapping
    """
    logger.info(f"Fitting Non-Homogeneous HMM (NHHMM) with {n_components} states...")
    
    detector = NHHMMRegimeDetector(
        n_states=n_components,
        n_iter=150,
        random_state=random_state
    )
    detector.fit(train_df)
    
    return detector, detector, detector.state_mapping


def assign_hmm_labels(df, detector, dummy_hmm=None, dummy_mapping=None):
    """
    Assign NHHMM regime labels, states, and confidence.
    """
    logger.info("Decoding regime states using NHHMM custom Viterbi...")
    return detector.assign_labels(df)


def save_hmm_model(detector, dummy_hmm=None, dummy_mapping=None, output_dir=None):
    """Save NHHMM detector object to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detector.save(output_dir)
