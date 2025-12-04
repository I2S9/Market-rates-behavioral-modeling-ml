"""
Time series preprocessing and cleaning module.

This module provides functions to clean, normalize, and prepare financial
time series data for modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Literal
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. ADF test will be skipped.")
    print("Install with: pip install statsmodels")


def load_rates_from_db(
    db_path: str = "data/database.db",
    series_names: Optional[list] = None
) -> pd.DataFrame:
    """
    Load interest rate series from SQLite database.
    
    Args:
        db_path: Path to SQLite database
        series_names: List of series names to load (e.g., ['DGS10', 'DFF', 'DGS30'])
                     If None, loads all available rate series
        
    Returns:
        DataFrame with date index and rate columns
    """
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.data.sql_ingestion import execute_query
    
    if series_names is None:
        series_names = ['DGS10', 'DFF', 'DGS30']
    
    dfs = []
    for series in series_names:
        try:
            query = f"SELECT date, {series} FROM {series} ORDER BY date"
            df = execute_query(query)
            if df is not None and len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df = df.rename(columns={series: series})
                dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {series}: {e}")
    
    if not dfs:
        raise ValueError("No series could be loaded from database")
    
    result = pd.concat(dfs, axis=1, join='outer')
    return result


def uniformize_frequency(
    df: pd.DataFrame,
    freq: str = 'D',
    method: str = 'ffill'
) -> pd.DataFrame:
    """
    Uniformize time series frequency.
    
    Args:
        df: DataFrame with datetime index
        freq: Target frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
        method: Method for resampling ('ffill', 'bfill', 'mean', 'last')
        
    Returns:
        DataFrame with uniform frequency
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    if method == 'ffill':
        resampled = df.resample(freq).ffill()
    elif method == 'bfill':
        resampled = df.resample(freq).bfill()
    elif method == 'mean':
        resampled = df.resample(freq).mean()
    elif method == 'last':
        resampled = df.resample(freq).last()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return resampled


def handle_missing_values(
    df: pd.DataFrame,
    method: str = 'interpolate',
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Handle missing values in time series.
    
    Args:
        df: DataFrame with potential missing values
        method: Method to use ('interpolate', 'ffill', 'bfill', 'drop')
        limit: Maximum number of consecutive missing values to fill
        
    Returns:
        DataFrame with missing values handled
    """
    df_clean = df.copy()
    
    if method == 'interpolate':
        df_clean = df_clean.interpolate(method='time', limit=limit)
    elif method == 'ffill':
        df_clean = df_clean.fillna(method='ffill', limit=limit)
    elif method == 'bfill':
        df_clean = df_clean.fillna(method='bfill', limit=limit)
    elif method == 'drop':
        df_clean = df_clean.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return df_clean


def test_stationarity(
    series: pd.Series,
    alpha: float = 0.05
) -> dict:
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    
    Args:
        series: Time series to test
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    if not HAS_STATSMODELS:
        return {
            'is_stationary': None,
            'adf_statistic': None,
            'p_value': None,
            'critical_values': None,
            'message': 'ADF test not available (statsmodels not installed)'
        }
    
    series_clean = series.dropna()
    
    if len(series_clean) < 10:
        return {
            'is_stationary': False,
            'adf_statistic': None,
            'p_value': None,
            'critical_values': None,
            'message': 'Insufficient data for ADF test'
        }
    
    result = adfuller(series_clean, autolag='AIC')
    
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]
    is_stationary = p_value < alpha
    
    return {
        'is_stationary': is_stationary,
        'adf_statistic': adf_statistic,
        'p_value': p_value,
        'critical_values': critical_values,
        'message': 'Stationary' if is_stationary else 'Non-stationary'
    }


def normalize_series(
    df: pd.DataFrame,
    method: Literal['min_max', 'z_score', 'none'] = 'none'
) -> tuple[pd.DataFrame, Optional[dict]]:
    """
    Normalize time series data.
    
    Args:
        df: DataFrame to normalize
        method: Normalization method ('min_max', 'z_score', 'none')
        
    Returns:
        Tuple of (normalized DataFrame, scaling parameters dict)
    """
    if method == 'none':
        return df, None
    
    df_norm = df.copy()
    scaling_params = {}
    
    for col in df.columns:
        series = df[col].dropna()
        
        if method == 'min_max':
            min_val = series.min()
            max_val = series.max()
            if max_val != min_val:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
                scaling_params[col] = {'min': min_val, 'max': max_val, 'method': 'min_max'}
            else:
                df_norm[col] = 0.5
                scaling_params[col] = {'min': min_val, 'max': max_val, 'method': 'min_max'}
        
        elif method == 'z_score':
            mean_val = series.mean()
            std_val = series.std()
            if std_val > 0:
                df_norm[col] = (df[col] - mean_val) / std_val
                scaling_params[col] = {'mean': mean_val, 'std': std_val, 'method': 'z_score'}
            else:
                df_norm[col] = 0
                scaling_params[col] = {'mean': mean_val, 'std': std_val, 'method': 'z_score'}
    
    return df_norm, scaling_params


def preprocess_rates(
    db_path: str = "data/database.db",
    series_names: Optional[list] = None,
    freq: str = 'D',
    missing_method: str = 'interpolate',
    normalize: bool = False,
    normalize_method: Literal['min_max', 'z_score'] = 'min_max',
    output_path: str = "data/processed/rates_clean.csv"
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for interest rate series.
    
    Args:
        db_path: Path to SQLite database
        series_names: List of series to process
        freq: Target frequency ('D', 'W', 'M')
        missing_method: Method for handling missing values
        normalize: Whether to normalize the data
        normalize_method: Normalization method if normalize=True
        output_path: Path to save cleaned data
        
    Returns:
        Cleaned DataFrame
    """
    print("Loading data from database...")
    df = load_rates_from_db(db_path, series_names)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} series")
    
    print(f"\nUniformizing frequency to {freq}...")
    df = uniformize_frequency(df, freq=freq, method='ffill')
    print(f"  After resampling: {len(df)} rows")
    
    print(f"\nHandling missing values using {missing_method}...")
    missing_before = df.isna().sum().sum()
    df = handle_missing_values(df, method=missing_method, limit=5)
    # Fill any remaining NaN with forward fill then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    missing_after = df.isna().sum().sum()
    print(f"  Missing values: {missing_before} -> {missing_after}")
    
    print("\nTesting stationarity...")
    for col in df.columns:
        result = test_stationarity(df[col])
        status = "✓" if result['is_stationary'] else "✗"
        p_value_str = f"{result['p_value']:.4f}" if result['p_value'] is not None else "N/A"
        print(f"  {status} {col}: {result['message']} (p-value: {p_value_str})")
    
    if normalize:
        print(f"\nNormalizing data using {normalize_method}...")
        df, scaling_params = normalize_series(df, method=normalize_method)
        print("  Normalization complete")
    else:
        scaling_params = None
    
    print(f"\nSaving cleaned data to {output_path}...")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"  Saved {len(df)} rows, {len(df.columns)} columns")
    
    if scaling_params:
        import json
        params_path = output_path.replace('.csv', '_scaling_params.json')
        with open(params_path, 'w') as f:
            json.dump(scaling_params, f, indent=2)
        print(f"  Saved scaling parameters to {params_path}")
    
    return df


if __name__ == "__main__":
    df_clean = preprocess_rates(
        series_names=['DGS10', 'DFF', 'DGS30'],
        freq='D',
        missing_method='interpolate',
        normalize=False
    )
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)
    print(f"\nFinal dataset shape: {df_clean.shape}")
    print(f"\nFirst few rows:")
    print(df_clean.head())
    print(f"\nLast few rows:")
    print(df_clean.tail())
    print(f"\nSummary statistics:")
    print(df_clean.describe())

