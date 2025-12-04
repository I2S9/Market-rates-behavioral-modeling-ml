"""
Advanced evaluation and monitoring module for model stability and drift detection.

This module provides ALM/Risk Management capabilities:
- Rolling RMSE over time
- Feature distribution drift detection
- Model stability index
- Dashboard-ready exports for Power BI
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from scipy import stats
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def calculate_rolling_rmse(
    y_true: pd.Series,
    y_pred: pd.Series,
    window_size: int = 30
) -> pd.Series:
    """
    Calculate rolling RMSE over time.
    
    Args:
        y_true: True values with datetime index
        y_pred: Predicted values
        window_size: Size of rolling window
        
    Returns:
        Series with rolling RMSE values
    """
    if not isinstance(y_true.index, pd.DatetimeIndex):
        raise ValueError("y_true must have a DatetimeIndex")
    
    errors = (y_true - y_pred) ** 2
    rolling_mse = errors.rolling(window=window_size, min_periods=1).mean()
    rolling_rmse = np.sqrt(rolling_mse)
    
    return rolling_rmse


def calculate_psi(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10
) -> float:
    """
    Calculate Population Stability Index (PSI) for drift detection.
    
    Args:
        expected: Expected distribution (training/reference)
        actual: Actual distribution (current/production)
        bins: Number of bins for discretization
        
    Returns:
        PSI value (lower is better, < 0.1 is stable, > 0.25 indicates significant drift)
    """
    # Remove NaN values
    expected_clean = expected.dropna()
    actual_clean = actual.dropna()
    
    if len(expected_clean) == 0 or len(actual_clean) == 0:
        return np.nan
    
    # Create bins based on expected distribution
    _, bin_edges = pd.cut(expected_clean, bins=bins, retbins=True, duplicates='drop')
    
    # Calculate expected and actual distributions
    expected_counts = pd.cut(expected_clean, bins=bin_edges, include_lowest=True).value_counts().sort_index()
    actual_counts = pd.cut(actual_clean, bins=bin_edges, include_lowest=True).value_counts().sort_index()
    
    # Normalize to percentages
    expected_pct = expected_counts / len(expected_clean)
    actual_pct = actual_counts / len(actual_clean)
    
    # Handle missing bins in actual
    for idx in expected_pct.index:
        if idx not in actual_pct.index:
            actual_pct[idx] = 0.0
    
    actual_pct = actual_pct.reindex(expected_pct.index, fill_value=0.0)
    
    # Calculate PSI
    psi = 0.0
    for idx in expected_pct.index:
        if expected_pct[idx] > 0:
            psi += (actual_pct[idx] - expected_pct[idx]) * np.log(actual_pct[idx] / expected_pct[idx] + 1e-10)
    
    return psi


def detect_feature_drift(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = 'psi',
    threshold: float = 0.25
) -> pd.DataFrame:
    """
    Detect drift in feature distributions between train and test sets.
    
    Args:
        X_train: Training features
        X_test: Test features
        method: Method for drift detection ('psi' or 'ks')
        threshold: Threshold for significant drift
        
    Returns:
        DataFrame with drift metrics per feature
    """
    drift_results = []
    
    for col in X_train.columns:
        if col not in X_test.columns:
            continue
        
        train_series = X_train[col].dropna()
        test_series = X_test[col].dropna()
        
        if len(train_series) == 0 or len(test_series) == 0:
            continue
        
        if method == 'psi':
            drift_score = calculate_psi(train_series, test_series)
            drift_status = 'Significant' if drift_score > threshold else 'Stable'
        elif method == 'ks':
            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(train_series, test_series)
            drift_score = ks_stat
            drift_status = 'Significant' if p_value < 0.05 else 'Stable'
        else:
            raise ValueError(f"Unknown method: {method}")
        
        drift_results.append({
            'feature': col,
            'drift_score': drift_score,
            'drift_status': drift_status,
            'train_mean': train_series.mean(),
            'test_mean': test_series.mean(),
            'mean_diff_pct': abs((test_series.mean() - train_series.mean()) / (train_series.mean() + 1e-10)) * 100
        })
    
    return pd.DataFrame(drift_results)


def calculate_model_stability_index(
    rolling_rmse: pd.Series,
    baseline_rmse: float
) -> float:
    """
    Calculate model stability index (0-1 scale, higher is better).
    
    Args:
        rolling_rmse: Rolling RMSE over time
        baseline_rmse: Baseline RMSE for comparison
        
    Returns:
        Stability index (0-1)
    """
    if len(rolling_rmse) == 0:
        return 0.0
    
    # Coefficient of variation of rolling RMSE (lower is more stable)
    cv = rolling_rmse.std() / (rolling_rmse.mean() + 1e-10)
    
    # Stability index: inverse of CV, normalized
    stability = 1 / (1 + cv)
    
    # Penalize if RMSE is significantly higher than baseline
    rmse_ratio = rolling_rmse.mean() / (baseline_rmse + 1e-10)
    if rmse_ratio > 1.5:
        stability *= 0.7
    elif rmse_ratio > 1.2:
        stability *= 0.85
    
    return min(stability, 1.0)


def create_monitoring_dashboard(
    predictions_path: str,
    features_path: str,
    target_col: str,
    model_name: str = "XGBoost",
    window_size: int = 30,
    output_dir: str = "dashboards"
) -> pd.DataFrame:
    """
    Create comprehensive monitoring dashboard data for Power BI.
    
    Args:
        predictions_path: Path to predictions CSV
        features_path: Path to features CSV
        target_col: Target column name
        model_name: Name of model to monitor
        window_size: Rolling window size
        output_dir: Directory to save dashboard data
        
    Returns:
        DataFrame with all monitoring metrics ready for Power BI
    """
    print("="*60)
    print(f"Creating Monitoring Dashboard for {target_col} - {model_name}")
    print("="*60)
    
    # Load predictions
    print(f"\nLoading predictions from {predictions_path}...")
    pred_df = pd.read_csv(predictions_path, index_col=0, parse_dates=True)
    
    # Get prediction column
    pred_col = None
    # Try exact match first
    model_lower = model_name.lower()
    for col in pred_df.columns:
        col_lower = col.lower()
        if model_lower in col_lower and 'pred' in col_lower:
            if 'actual' not in col_lower:
                pred_col = col
                break
    
    # If not found, try partial match
    if pred_col is None:
        for col in pred_df.columns:
            col_lower = col.lower()
            if 'pred' in col_lower and 'actual' not in col_lower:
                # Check if model name matches (xgboost -> xgb, randomforest -> random)
                if (model_lower == 'xgboost' and 'xgb' in col_lower) or \
                   (model_lower == 'randomforest' and 'random' in col_lower) or \
                   (model_lower == 'gradientboosting' and 'gradient' in col_lower):
                    pred_col = col
                    break
    
    # Last resort: take first prediction column
    if pred_col is None:
        for col in pred_df.columns:
            if 'pred' in col.lower() and 'actual' not in col.lower():
                pred_col = col
                break
    
    if pred_col is None:
        raise ValueError(f"Could not find prediction column for {model_name}")
    
    print(f"  Found prediction column: {pred_col}")
    print(f"  Date range: {pred_df.index.min()} to {pred_df.index.max()}")
    
    # Calculate rolling RMSE
    print(f"\nCalculating rolling RMSE (window={window_size})...")
    y_true = pred_df['actual']
    y_pred = pred_df[pred_col]
    
    rolling_rmse = calculate_rolling_rmse(y_true, y_pred, window_size=window_size)
    overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"  Overall RMSE: {overall_rmse:.4f}")
    print(f"  Mean rolling RMSE: {rolling_rmse.mean():.4f}")
    print(f"  Rolling RMSE std: {rolling_rmse.std():.4f}")
    
    # Load features for drift detection
    print(f"\nLoading features for drift detection...")
    features_df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    
    # Split features into train and test (temporal split)
    split_idx = int(len(features_df) * 0.8)
    X_train = features_df.iloc[:split_idx]
    X_test = features_df.iloc[split_idx:]
    
    # Detect feature drift
    print(f"\nDetecting feature drift...")
    drift_df = detect_feature_drift(X_train, X_test, method='psi')
    
    significant_drift = drift_df[drift_df['drift_status'] == 'Significant']
    print(f"  Features with significant drift: {len(significant_drift)}/{len(drift_df)}")
    if len(significant_drift) > 0:
        print(f"  Top drifted features:")
        for _, row in significant_drift.nlargest(5, 'drift_score').iterrows():
            print(f"    {row['feature']}: PSI={row['drift_score']:.4f}")
    
    # Calculate model stability index
    print(f"\nCalculating model stability index...")
    stability_index = calculate_model_stability_index(rolling_rmse, overall_rmse)
    print(f"  Stability index: {stability_index:.4f} (0-1 scale, higher is better)")
    
    # Create dashboard DataFrame
    print(f"\nCreating dashboard data...")
    dashboard_data = pd.DataFrame({
        'target': target_col,
        'model': model_name,
        'actual': y_true.values,
        'predicted': y_pred.values,
        'error': (y_true - y_pred).values,
        'error_abs': abs(y_true - y_pred).values,
        'rolling_rmse': rolling_rmse.values,
        'stability_index': stability_index,
        'overall_rmse': overall_rmse
    }, index=pred_df.index)
    
    # Add date column (not as index to avoid duplication)
    dashboard_data['date'] = dashboard_data.index
    
    # Add monthly aggregations for Power BI
    dashboard_data['year'] = dashboard_data.index.year
    dashboard_data['month'] = dashboard_data.index.month
    dashboard_data['quarter'] = dashboard_data.index.quarter
    dashboard_data['year_month'] = dashboard_data.index.to_period('M').astype(str)
    
    # Save dashboard data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dashboard_file = output_path / f"{target_col}_{model_name.lower()}_monitoring.csv"
    dashboard_data.to_csv(dashboard_file, index=False)
    print(f"  Saved dashboard data to {dashboard_file}")
    
    # Save drift report
    drift_file = output_path / f"{target_col}_{model_name.lower()}_drift_report.csv"
    drift_df.to_csv(drift_file, index=False)
    print(f"  Saved drift report to {drift_file}")
    
    # Create summary metrics
    summary_metrics = {
        'target': target_col,
        'model': model_name,
        'overall_rmse': overall_rmse,
        'mean_rolling_rmse': rolling_rmse.mean(),
        'std_rolling_rmse': rolling_rmse.std(),
        'stability_index': stability_index,
        'features_with_drift': len(significant_drift),
        'total_features': len(drift_df),
        'drift_rate': len(significant_drift) / len(drift_df) if len(drift_df) > 0 else 0.0
    }
    
    import json
    summary_file = output_path / f"{target_col}_{model_name.lower()}_monitoring_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"  Saved summary metrics to {summary_file}")
    
    print(f"\n{'='*60}")
    print("Monitoring Dashboard Created")
    print(f"{'='*60}")
    
    return dashboard_data


def create_all_monitoring_dashboards(
    predictions_dir: str = "reports",
    features_path: str = "data/processed/rates_features.csv",
    target_columns: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    output_dir: str = "dashboards"
) -> Dict:
    """
    Create monitoring dashboards for all models and targets.
    
    Args:
        predictions_dir: Directory containing prediction CSV files
        features_path: Path to features CSV
        target_columns: List of target columns
        model_names: List of model names to monitor
        output_dir: Directory to save dashboard data
        
    Returns:
        Dictionary with dashboard dataframes
    """
    if target_columns is None:
        target_columns = ['DGS10', 'DFF', 'DGS30']
    
    if model_names is None:
        model_names = ['XGBoost', 'RandomForest']
    
    results = {}
    pred_dir = Path(predictions_dir)
    
    for target_col in target_columns:
        for model_name in model_names:
            # Find prediction file
            pred_file = pred_dir / f"{target_col}_ml_predictions.csv"
            
            if not pred_file.exists():
                print(f"\nWarning: {pred_file} not found, skipping...")
                continue
            
            try:
                dashboard_df = create_monitoring_dashboard(
                    str(pred_file),
                    features_path,
                    target_col,
                    model_name,
                    window_size=30,
                    output_dir=output_dir
                )
                results[f"{target_col}_{model_name}"] = dashboard_df
            except Exception as e:
                print(f"\nError creating dashboard for {target_col} - {model_name}: {e}")
    
    return results


if __name__ == "__main__":
    results = create_all_monitoring_dashboards(
        target_columns=['DGS10', 'DFF', 'DGS30'],
        model_names=['XGBoost', 'RandomForest']
    )
    
    print("\nSummary:")
    for key, df in results.items():
        print(f"\n{key}:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Stability index: {df['stability_index'].iloc[0]:.4f}")

