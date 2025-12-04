"""
Modeling modules for forecasting and behavioral modeling.

This package provides:
- Baseline statistical models (ARIMA, Prophet)
- Machine learning models
- Model evaluation utilities
"""

from .baseline_models import (
    calculate_metrics,
    train_test_split_ts,
    fit_arima,
    predict_arima,
    fit_prophet,
    predict_prophet,
    train_baseline_models
)

__all__ = [
    'calculate_metrics',
    'train_test_split_ts',
    'fit_arima',
    'predict_arima',
    'fit_prophet',
    'predict_prophet',
    'train_baseline_models'
]

