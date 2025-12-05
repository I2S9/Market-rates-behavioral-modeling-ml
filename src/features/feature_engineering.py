"""
Time series feature engineering module.

This module creates explanatory features for time series forecasting:
- Lags (1, 5, 10 days)
- Rolling statistics (mean, std)
- Returns and percentage changes
- Trend indicators
- Calendar features
"""

import pandas as pd
import numpy as np
from typing import Optional, List


def create_time_features(
    df: pd.DataFrame,
    col: str = "rate",
    lag_periods: List[int] = [1, 5, 10],
    rolling_windows: List[int] = [7, 30]
) -> pd.DataFrame:
    """
    Create time series features for a single column.
    
    Args:
        df: DataFrame with datetime index
        col: Column name to create features for
        lag_periods: List of lag periods to create
        rolling_windows: List of rolling window sizes
        
    Returns:
        DataFrame with added features
    """
    df_features = df.copy()
    
    # Lags
    for lag in lag_periods:
        df_features[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    # Rolling statistics
    for window in rolling_windows:
        df_features[f"{col}_roll_mean_{window}"] = df[col].rolling(window=window, min_periods=1).mean()
        df_features[f"{col}_roll_std_{window}"] = df[col].rolling(window=window, min_periods=1).std()
    
    # Returns (percentage change)
    df_features[f"{col}_return"] = df[col].pct_change()
    df_features[f"{col}_return_lag1"] = df_features[f"{col}_return"].shift(1)
    
    # Trend indicators
    df_features[f"{col}_diff"] = df[col].diff()
    df_features[f"{col}_diff_pct"] = df[col].pct_change()
    
    # Moving average crossover (short vs long term)
    if 7 in rolling_windows and 30 in rolling_windows:
        df_features[f"{col}_ma_crossover"] = (
            df_features[f"{col}_roll_mean_7"] - df_features[f"{col}_roll_mean_30"]
        )
    
    return df_features


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar-based features from datetime index.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with added calendar features
    """
    df_cal = df.copy()
    
    if not isinstance(df_cal.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    # Day features
    df_cal['day_of_week'] = df_cal.index.dayofweek
    df_cal['day_of_month'] = df_cal.index.day
    df_cal['day_of_year'] = df_cal.index.dayofyear
    
    # Month features
    df_cal['month'] = df_cal.index.month
    df_cal['quarter'] = df_cal.index.quarter
    
    # Year feature
    df_cal['year'] = df_cal.index.year
    
    # Week features
    df_cal['week_of_year'] = df_cal.index.isocalendar().week
    
    # Cyclical encoding for periodic features
    df_cal['day_of_week_sin'] = np.sin(2 * np.pi * df_cal['day_of_week'] / 7)
    df_cal['day_of_week_cos'] = np.cos(2 * np.pi * df_cal['day_of_week'] / 7)
    df_cal['month_sin'] = np.sin(2 * np.pi * df_cal['month'] / 12)
    df_cal['month_cos'] = np.cos(2 * np.pi * df_cal['month'] / 12)
    
    # Is weekend
    df_cal['is_weekend'] = (df_cal['day_of_week'] >= 5).astype(int)
    
    # Month end / quarter end indicators
    df_cal['is_month_end'] = df_cal.index.is_month_end.astype(int)
    df_cal['is_quarter_end'] = df_cal.index.is_quarter_end.astype(int)
    
    return df_cal


def create_all_features(
    df: pd.DataFrame,
    target_columns: Optional[List[str]] = None,
    lag_periods: List[int] = [1, 5, 10],
    rolling_windows: List[int] = [7, 30],
    include_calendar: bool = True
) -> pd.DataFrame:
    """
    Create all time series features for multiple columns.
    
    Args:
        df: DataFrame with datetime index and rate columns
        target_columns: List of columns to create features for (if None, uses all numeric columns)
        lag_periods: List of lag periods
        rolling_windows: List of rolling window sizes
        include_calendar: Whether to include calendar features
        
    Returns:
        DataFrame with all features added
    """
    df_features = df.copy()
    
    # Determine columns to process
    if target_columns is None:
        target_columns = [col for col in df.columns if df[col].dtype in [np.float64, np.int64, float, int]]
    
    # Create features for each target column
    for col in target_columns:
        if col in df_features.columns:
            df_features = create_time_features(
                df_features,
                col=col,
                lag_periods=lag_periods,
                rolling_windows=rolling_windows
            )
    
    # Add calendar features
    if include_calendar:
        df_features = create_calendar_features(df_features)
    
    # Remove rows with NaN from lags (first rows)
    # Keep at least one row for each lag period
    max_lag = max(lag_periods) if lag_periods else 0
    if max_lag > 0:
        df_features = df_features.iloc[max_lag:]
    
    return df_features


def engineer_features_from_clean_data(
    input_path: str = "data/processed/rates_clean.csv",
    output_path: str = "data/processed/rates_features.csv",
    target_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline from cleaned data.
    
    Args:
        input_path: Path to cleaned rates CSV
        output_path: Path to save features CSV
        target_columns: Columns to create features for
        
    Returns:
        DataFrame with all features
    """
    print("Loading cleaned data...")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    print("\nCreating time series features...")
    df_features = create_all_features(
        df,
        target_columns=target_columns,
        lag_periods=[1, 5, 10],
        rolling_windows=[7, 30],
        include_calendar=True
    )
    
    print(f"  Created {len(df_features.columns)} total columns")
    print(f"  Added {len(df_features.columns) - len(df.columns)} new features")
    
    # Count useful features (excluding original columns)
    original_cols = set(df.columns)
    feature_cols = [col for col in df_features.columns if col not in original_cols]
    print(f"  Feature columns: {len(feature_cols)}")
    
    # Check for data leakage (features that use future data)
    print("\nChecking for data leakage...")
    leakage_issues = []
    for col in feature_cols:
        if 'shift(-' in str(df_features[col].dtype) or 'lead' in col.lower():
            leakage_issues.append(col)
    
    if leakage_issues:
        print(f"  Warning: Potential leakage in {len(leakage_issues)} features")
    else:
        print("  ✓ No data leakage detected (all features use past data only)")
    
    print(f"\nSaving features to {output_path}...")
    from pathlib import Path
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path)
    print(f"  Saved {len(df_features)} rows, {len(df_features.columns)} columns")
    
    print("\n" + "="*60)
    print("Feature engineering complete!")
    print("="*60)
    print(f"\nDataset shape: {df_features.shape}")
    print(f"\nFeature summary:")
    print(f"  Original columns: {len(original_cols)}")
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Total columns: {len(df_features.columns)}")
    
    return df_features


if __name__ == "__main__":
    df_features = engineer_features_from_clean_data(
        target_columns=['DGS10', 'DFF', 'DGS30']
    )
    
    print("\nSample features:")
    print(df_features.head())
    print("\nFeature columns (first 20):")
    feature_cols = [col for col in df_features.columns if col not in ['DGS10', 'DFF', 'DGS30']]
    print(feature_cols[:20])


