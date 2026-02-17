#!/usr/bin/env python3
"""
RUL Labeling Engine (Answer Key)
================================
Generates training targets (Remaining Useful Life) for the AI model.

Logic:
1. Load feature data.
2. Identify the last timestamp as "failure" (RUL = 0).
3. Calculate RUL for all preceding rows.
4. Merge RUL back into the dataset.
5. Check correlations to validate physics.

Usage:
    python labeling_engine.py
"""

import sys
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "features.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "labeled_features.csv"

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("labeler")

# =============================================================================
# CORE LOGIC
# =============================================================================

def calculate_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Remaining Useful Life (RUL) in minutes.
    Assumes the dataset represents a single run-to-failure trajectory.
    """
    # Ensure timestamps are datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by time
    df = df.sort_values(by='timestamp')
    
    # Identify failure time (last timestamp)
    failure_time = df['timestamp'].iloc[-1]
    logger.info("Identified Failure Time: %s", failure_time)
    
    # Calculate RUL
    # RUL = Failure Time - Current Time
    df['rul_minutes'] = (failure_time - df['timestamp']).dt.total_seconds() / 60
    
    return df

def validate_correlations(df: pd.DataFrame):
    """
    Checks if features correlate with RUL.
    Expectation: As RUL decreases, degradation features (RMS, Kurtosis) should INCREASE.
    This implies a NEGATIVE correlation.
    """
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=['number'])
    
    if 'rul_minutes' not in numeric_df.columns:
        logger.warning("RUL column missing, cannot calculate correlations.")
        return

    correlations = numeric_df.corr()['rul_minutes'].sort_values()
    
    logger.info("--- Feature vs RUL Correlations ---")
    logger.info("(Negative values indicate feature rises as RUL drops -> Good predictor)")
    for feature, corr in correlations.items():
        if feature != 'rul_minutes':
            logger.info(f"{feature.ljust(15)}: {corr:.4f}")

# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    if not INPUT_FILE.exists():
        logger.error("Input file not found: %s", INPUT_FILE)
        sys.exit(1)

    logger.info("Loading features from %s...", INPUT_FILE)
    df = pd.read_csv(INPUT_FILE)
    
    if df.empty:
        logger.error("Input file is empty.")
        sys.exit(1)

    # 1. Calculate RUL
    labeled_df = calculate_rul(df)
    
    # 2. Validate
    validate_correlations(labeled_df)
    
    # 3. Save
    labeled_df.to_csv(OUTPUT_FILE, index=False)
    logger.info("Labeled data saved to %s", OUTPUT_FILE)
    logger.info("Rows processed: %d", len(labeled_df))
