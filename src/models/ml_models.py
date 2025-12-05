"""
Machine Learning models for time series forecasting.

This module implements Random Forest and Gradient Boosting models
to compare with baseline statistical models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import json
warnings.filterwarnings('ignore')

# XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: xgboost not installed. XGBoost will be unavailable.")
    print("Install with: pip install xgboost")

# LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: lightgbm not installed. LightGBM will be unavailable.")
    print("Install with: pip install lightgbm")


def prepare_ml_data(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for ML models.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        exclude_cols: Columns to exclude from features
        
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Exclude target and other rate columns from features
    feature_cols = [col for col in df.columns 
                    if col != target_col 
                    and col not in exclude_cols
                    and col not in ['DGS10', 'DFF', 'DGS30']]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Remove rows with NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    return X, y


def train_test_split_ts_ml(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split time series data for ML (chronological, no shuffle).
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of data for testing
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    
    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    random_state: int = 42
) -> Tuple[RandomForestRegressor, Dict]:
    """
    Train Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed
        
    Returns:
        Tuple of (fitted model, model info)
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    model_info = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'n_features': X_train.shape[1]
    }
    
    return model, model_info


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42
) -> Tuple[Optional[object], Dict]:
    """
    Train XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        random_state: Random seed
        
    Returns:
        Tuple of (fitted model, model info)
    """
    if not HAS_XGBOOST:
        return None, {'error': 'xgboost not installed'}
    
    try:
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        model_info = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_features': X_train.shape[1]
        }
        
        return model, model_info
    
    except Exception as e:
        return None, {'error': str(e)}


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42
) -> Tuple[Optional[object], Dict]:
    """
    Train LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        random_state: Random seed
        
    Returns:
        Tuple of (fitted model, model info)
    """
    if not HAS_LIGHTGBM:
        return None, {'error': 'lightgbm not installed'}
    
    try:
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        model_info = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_features': X_train.shape[1]
        }
        
        return model, model_info
    
    except Exception as e:
        return None, {'error': str(e)}


def get_feature_importance(
    model: object,
    feature_names: List[str],
    model_type: str = 'random_forest'
) -> pd.DataFrame:
    """
    Extract feature importance from model.
    
    Args:
        model: Fitted model
        feature_names: List of feature names
        model_type: Type of model ('random_forest', 'xgboost', 'lightgbm')
        
    Returns:
        DataFrame with feature names and importance scores
    """
    if model_type == 'random_forest':
        importances = model.feature_importances_
    elif model_type in ['xgboost', 'lightgbm']:
        importances = model.feature_importances_
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Normalize to percentage
    df_importance['importance_pct'] = (df_importance['importance'] / df_importance['importance'].sum()) * 100
    
    return df_importance


def calculate_metrics_ml(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate RMSE and MAE metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with RMSE and MAE
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae
    }


def train_ml_models(
    features_path: str = "data/processed/rates_features.csv",
    target_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    baseline_metrics_path: str = "reports",
    output_dir: str = "reports"
) -> Dict:
    """
    Train ML models and compare with baseline.
    
    Args:
        features_path: Path to features CSV
        target_columns: List of target columns to forecast
        test_size: Proportion of data for testing
        baseline_metrics_path: Path to baseline metrics JSON files
        output_dir: Directory to save results
        
    Returns:
        Dictionary with model results
    """
    print("="*60)
    print("Training Machine Learning Models")
    print("="*60)
    
    # Load features
    print(f"\nLoading features from {features_path}...")
    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    if target_columns is None:
        target_columns = ['DGS10', 'DFF', 'DGS30']
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for target_col in target_columns:
        if target_col not in df.columns:
            print(f"\nWarning: {target_col} not found in data, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Forecasting {target_col}")
        print(f"{'='*60}")
        
        # Prepare data
        X, y = prepare_ml_data(df, target_col, exclude_cols=target_columns)
        print(f"\nFeatures: {X.shape[1]} features, {len(X)} samples")
        
        # Split data (temporal, no shuffle)
        X_train, X_test, y_train, y_test = train_test_split_ts_ml(X, y, test_size=test_size)
        print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
        
        col_results = {}
        
        # Load baseline metrics for comparison
        baseline_file = Path(baseline_metrics_path) / f"{target_col}_metrics.json"
        baseline_metrics = {}
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
                # Get best baseline (lowest RMSE)
                best_baseline = min(baseline_data.items(), key=lambda x: x[1]['RMSE'])
                baseline_metrics = {
                    'best_model': best_baseline[0],
                    'RMSE': best_baseline[1]['RMSE'],
                    'MAE': best_baseline[1]['MAE']
                }
                print(f"\nBaseline metrics: {baseline_metrics['best_model']} - RMSE={baseline_metrics['RMSE']:.4f}, MAE={baseline_metrics['MAE']:.4f}")
        
        # Random Forest
        print(f"\n--- Random Forest ---")
        rf_model, rf_info = train_random_forest(X_train, y_train, n_estimators=300)
        y_pred_rf = rf_model.predict(X_test)
        metrics_rf = calculate_metrics_ml(y_test, y_pred_rf)
        print(f"  RMSE: {metrics_rf['RMSE']:.4f}")
        print(f"  MAE: {metrics_rf['MAE']:.4f}")
        
        if baseline_metrics:
            improvement = ((baseline_metrics['RMSE'] - metrics_rf['RMSE']) / baseline_metrics['RMSE']) * 100
            print(f"  vs Baseline: {improvement:+.2f}% RMSE improvement")
        
        # Feature importance
        feature_importance_rf = get_feature_importance(rf_model, X_train.columns.tolist(), 'random_forest')
        
        col_results['RandomForest'] = {
            'model': rf_model,
            'predictions': y_pred_rf,
            'metrics': metrics_rf,
            'model_info': rf_info,
            'feature_importance': feature_importance_rf
        }
        
        # XGBoost or LightGBM
        if HAS_XGBOOST:
            print(f"\n--- XGBoost ---")
            xgb_model, xgb_info = train_xgboost(X_train, y_train, n_estimators=300)
            
            if xgb_model is not None:
                y_pred_xgb = xgb_model.predict(X_test)
                metrics_xgb = calculate_metrics_ml(y_test, y_pred_xgb)
                print(f"  RMSE: {metrics_xgb['RMSE']:.4f}")
                print(f"  MAE: {metrics_xgb['MAE']:.4f}")
                
                if baseline_metrics:
                    improvement = ((baseline_metrics['RMSE'] - metrics_xgb['RMSE']) / baseline_metrics['RMSE']) * 100
                    print(f"  vs Baseline: {improvement:+.2f}% RMSE improvement")
                
                feature_importance_xgb = get_feature_importance(xgb_model, X_train.columns.tolist(), 'xgboost')
                
                col_results['XGBoost'] = {
                    'model': xgb_model,
                    'predictions': y_pred_xgb,
                    'metrics': metrics_xgb,
                    'model_info': xgb_info,
                    'feature_importance': feature_importance_xgb
                }
            else:
                print(f"  Error: {xgb_info.get('error', 'Unknown error')}")
        
        elif HAS_LIGHTGBM:
            print(f"\n--- LightGBM ---")
            lgb_model, lgb_info = train_lightgbm(X_train, y_train, n_estimators=300)
            
            if lgb_model is not None:
                y_pred_lgb = lgb_model.predict(X_test)
                metrics_lgb = calculate_metrics_ml(y_test, y_pred_lgb)
                print(f"  RMSE: {metrics_lgb['RMSE']:.4f}")
                print(f"  MAE: {metrics_lgb['MAE']:.4f}")
                
                if baseline_metrics:
                    improvement = ((baseline_metrics['RMSE'] - metrics_lgb['RMSE']) / baseline_metrics['RMSE']) * 100
                    print(f"  vs Baseline: {improvement:+.2f}% RMSE improvement")
                
                feature_importance_lgb = get_feature_importance(lgb_model, X_train.columns.tolist(), 'lightgbm')
                
                col_results['LightGBM'] = {
                    'model': lgb_model,
                    'predictions': y_pred_lgb,
                    'metrics': metrics_lgb,
                    'model_info': lgb_info,
                    'feature_importance': feature_importance_lgb
                }
            else:
                print(f"  Error: {lgb_info.get('error', 'Unknown error')}")
        else:
            print(f"\n--- Gradient Boosting ---")
            print("  XGBoost and LightGBM not available")
        
        # Save predictions
        print(f"\n--- Saving Results ---")
        predictions_df = pd.DataFrame({
            'date': y_test.index,
            'actual': y_test.values
        }, index=y_test.index)
        
        if 'RandomForest' in col_results:
            predictions_df['random_forest_pred'] = col_results['RandomForest']['predictions']
        
        if 'XGBoost' in col_results:
            predictions_df['xgboost_pred'] = col_results['XGBoost']['predictions']
        elif 'LightGBM' in col_results:
            predictions_df['lightgbm_pred'] = col_results['LightGBM']['predictions']
        
        pred_file = output_path / f"{target_col}_ml_predictions.csv"
        predictions_df.to_csv(pred_file)
        print(f"  Saved predictions to {pred_file}")
        
        # Save metrics
        metrics_summary = {}
        for model_name, model_data in col_results.items():
            metrics_summary[model_name] = model_data['metrics']
        
        metrics_file = output_path / f"{target_col}_ml_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"  Saved metrics to {metrics_file}")
        
        # Save feature importance for Power BI
        for model_name, model_data in col_results.items():
            if 'feature_importance' in model_data:
                importance_df = model_data['feature_importance']
                importance_file = output_path / f"{target_col}_{model_name.lower()}_feature_importance.csv"
                importance_df.to_csv(importance_file, index=False)
                print(f"  Saved {model_name} feature importance to {importance_file}")
        
        results[target_col] = col_results
    
    # Summary
    print(f"\n{'='*60}")
    print("ML Models Training Complete")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    results = train_ml_models(
        target_columns=['DGS10', 'DFF', 'DGS30'],
        test_size=0.2
    )
    
    print("\nSummary of Results:")
    for target, models in results.items():
        print(f"\n{target}:")
        for model_name, model_data in models.items():
            metrics = model_data['metrics']
            print(f"  {model_name}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")


