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

from .ml_models import (
    prepare_ml_data,
    train_test_split_ts_ml,
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    get_feature_importance,
    calculate_metrics_ml,
    train_ml_models
)

from .lstm_model import (
    LSTMModel,
    TimeSeriesDataset,
    create_sequences,
    prepare_lstm_data,
    train_lstm,
    predict_lstm,
    plot_forecast_vs_actual,
    train_lstm_models
)

from .behavioral_models import (
    prepare_behavioral_data,
    train_logistic_regression,
    train_random_forest_classifier,
    train_gradient_boosting_classifier,
    evaluate_classifier,
    plot_roc_curves,
    plot_pr_curves,
    plot_score_distributions,
    train_behavioral_models
)

__all__ = [
    'calculate_metrics',
    'train_test_split_ts',
    'fit_arima',
    'predict_arima',
    'fit_prophet',
    'predict_prophet',
    'train_baseline_models',
    'prepare_ml_data',
    'train_test_split_ts_ml',
    'train_random_forest',
    'train_xgboost',
    'train_lightgbm',
    'get_feature_importance',
    'calculate_metrics_ml',
    'train_ml_models',
    'LSTMModel',
    'TimeSeriesDataset',
    'create_sequences',
    'prepare_lstm_data',
    'train_lstm',
    'predict_lstm',
    'plot_forecast_vs_actual',
    'train_lstm_models',
    'prepare_behavioral_data',
    'train_logistic_regression',
    'train_random_forest_classifier',
    'train_gradient_boosting_classifier',
    'evaluate_classifier',
    'plot_roc_curves',
    'plot_pr_curves',
    'plot_score_distributions',
    'train_behavioral_models'
]

