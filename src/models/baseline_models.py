"""
Baseline statistical forecasting models (ARIMA and Prophet).

This module implements baseline models for time series forecasting
to compare against machine learning models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ARIMA
try:
    from pmdarima import auto_arima
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False
    print("Warning: pmdarima not installed. ARIMA will be unavailable.")
    print("Install with: pip install pmdarima")

# Prophet
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("Warning: prophet not installed. Prophet will be unavailable.")
    print("Install with: pip install prophet")


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
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


def train_test_split_ts(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split time series data into train and test sets (chronological split).
    
    Args:
        df: DataFrame with datetime index
        target_col: Name of target column
        test_size: Proportion of data for testing
        
    Returns:
        Tuple of (train_df, test_df, y_train, y_test)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    return train_df, test_df, y_train, y_test


def fit_arima(
    y_train: pd.Series,
    max_p: int = 5,
    max_d: int = 2,
    max_q: int = 5,
    seasonal: bool = True,
    m: int = 12
) -> Tuple[Optional[object], Optional[Dict]]:
    """
    Fit ARIMA model using auto_arima.
    
    Args:
        y_train: Training time series
        max_p: Maximum AR order
        max_d: Maximum differencing order
        max_q: Maximum MA order
        seasonal: Whether to include seasonal component
        m: Seasonal period
        
    Returns:
        Tuple of (fitted model, model info dict)
    """
    if not HAS_PMDARIMA:
        return None, {'error': 'pmdarima not installed'}
    
    try:
        model = auto_arima(
            y_train,
            max_p=max_p,
            max_d=max_d,
            max_q=max_q,
            seasonal=seasonal,
            m=m,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        
        model_info = {
            'order': model.order,
            'seasonal_order': model.seasonal_order if seasonal else None,
            'aic': model.aic()
        }
        
        return model, model_info
    
    except Exception as e:
        return None, {'error': str(e)}


def predict_arima(
    model: object,
    n_periods: int
) -> np.ndarray:
    """
    Generate forecasts with ARIMA model.
    
    Args:
        model: Fitted ARIMA model
        n_periods: Number of periods to forecast
        
    Returns:
        Array of predictions
    """
    if model is None:
        return None
    
    forecast = model.predict(n_periods=n_periods)
    return forecast


def fit_prophet(
    y_train: pd.Series,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False
) -> Tuple[Optional[object], Optional[Dict]]:
    """
    Fit Prophet model.
    
    Args:
        y_train: Training time series with datetime index
        yearly_seasonality: Include yearly seasonality
        weekly_seasonality: Include weekly seasonality
        daily_seasonality: Include daily seasonality
        
    Returns:
        Tuple of (fitted model, model info dict)
    """
    if not HAS_PROPHET:
        return None, {'error': 'prophet not installed'}
    
    try:
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        df_prophet = pd.DataFrame({
            'ds': y_train.index,
            'y': y_train.values
        })
        
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        
        model.fit(df_prophet)
        
        model_info = {
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'daily_seasonality': daily_seasonality
        }
        
        return model, model_info
    
    except Exception as e:
        return None, {'error': str(e)}


def predict_prophet(
    model: object,
    periods: int,
    freq: str = 'D'
) -> pd.DataFrame:
    """
    Generate forecasts with Prophet model.
    
    Args:
        model: Fitted Prophet model
        periods: Number of periods to forecast
        freq: Frequency string ('D' for daily)
        
    Returns:
        DataFrame with 'ds' and 'yhat' columns
    """
    if model is None:
        return None
    
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    
    # Return only the forecasted periods
    return forecast.tail(periods)


def train_baseline_models(
    data_path: str = "data/processed/rates_clean.csv",
    target_columns: Optional[list] = None,
    test_size: float = 0.2,
    output_dir: str = "reports"
) -> Dict:
    """
    Train ARIMA and Prophet models for baseline forecasting.
    
    Args:
        data_path: Path to cleaned rates data
        target_columns: List of columns to forecast (if None, uses all numeric columns)
        test_size: Proportion of data for testing
        output_dir: Directory to save predictions and metrics
        
    Returns:
        Dictionary with model results and metrics
    """
    print("="*60)
    print("Training Baseline Models (ARIMA & Prophet)")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    if target_columns is None:
        target_columns = [col for col in df.columns if df[col].dtype in [np.float64, np.int64, float, int]]
    
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
        
        # Split data
        train_df, test_df, y_train, y_test = train_test_split_ts(
            df, target_col, test_size=test_size
        )
        print(f"\nTrain size: {len(y_train)}, Test size: {len(y_test)}")
        
        col_results = {}
        
        # ARIMA Model
        print(f"\n--- ARIMA Model ---")
        if HAS_PMDARIMA:
            arima_model, arima_info = fit_arima(y_train)
            
            if arima_model is not None:
                print(f"  ARIMA order: {arima_info.get('order')}")
                print(f"  AIC: {arima_info.get('aic', 'N/A'):.2f}")
                
                # Forecast
                y_pred_arima = predict_arima(arima_model, n_periods=len(y_test))
                
                if y_pred_arima is not None:
                    # Calculate metrics
                    metrics_arima = calculate_metrics(y_test, y_pred_arima)
                    print(f"  RMSE: {metrics_arima['RMSE']:.4f}")
                    print(f"  MAE: {metrics_arima['MAE']:.4f}")
                    
                    col_results['ARIMA'] = {
                        'model': arima_model,
                        'predictions': y_pred_arima,
                        'metrics': metrics_arima,
                        'model_info': arima_info
                    }
                else:
                    print("  Error generating ARIMA predictions")
            else:
                print(f"  Error fitting ARIMA: {arima_info.get('error', 'Unknown error')}")
        else:
            print("  ARIMA unavailable (pmdarima not installed)")
        
        # Prophet Model
        print(f"\n--- Prophet Model ---")
        if HAS_PROPHET:
            prophet_model, prophet_info = fit_prophet(y_train)
            
            if prophet_model is not None:
                print(f"  Prophet model fitted successfully")
                
                # Forecast
                forecast_df = predict_prophet(prophet_model, periods=len(y_test))
                
                if forecast_df is not None:
                    y_pred_prophet = forecast_df['yhat'].values
                    
                    # Calculate metrics
                    metrics_prophet = calculate_metrics(y_test, y_pred_prophet)
                    print(f"  RMSE: {metrics_prophet['RMSE']:.4f}")
                    print(f"  MAE: {metrics_prophet['MAE']:.4f}")
                    
                    col_results['Prophet'] = {
                        'model': prophet_model,
                        'predictions': y_pred_prophet,
                        'metrics': metrics_prophet,
                        'model_info': prophet_info
                    }
                else:
                    print("  Error generating Prophet predictions")
            else:
                print(f"  Error fitting Prophet: {prophet_info.get('error', 'Unknown error')}")
        else:
            print("  Prophet unavailable (prophet not installed)")
        
        # Save predictions
        print(f"\n--- Saving Predictions ---")
        predictions_df = pd.DataFrame({
            'date': y_test.index,
            'actual': y_test.values
        }, index=y_test.index)
        
        if 'ARIMA' in col_results:
            predictions_df['arima_pred'] = col_results['ARIMA']['predictions']
        
        if 'Prophet' in col_results:
            predictions_df['prophet_pred'] = col_results['Prophet']['predictions']
        
        pred_file = output_path / f"{target_col}_predictions.csv"
        predictions_df.to_csv(pred_file)
        print(f"  Saved predictions to {pred_file}")
        
        # Save metrics
        metrics_summary = {}
        if 'ARIMA' in col_results:
            metrics_summary['ARIMA'] = col_results['ARIMA']['metrics']
        if 'Prophet' in col_results:
            metrics_summary['Prophet'] = col_results['Prophet']['metrics']
        
        import json
        metrics_file = output_path / f"{target_col}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"  Saved metrics to {metrics_file}")
        
        results[target_col] = col_results
    
    # Summary
    print(f"\n{'='*60}")
    print("Baseline Models Training Complete")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    results = train_baseline_models(
        target_columns=['DGS10', 'DFF', 'DGS30'],
        test_size=0.2
    )
    
    print("\nSummary of Results:")
    for target, models in results.items():
        print(f"\n{target}:")
        for model_name, model_data in models.items():
            metrics = model_data['metrics']
            print(f"  {model_name}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")

