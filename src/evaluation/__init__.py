"""
Evaluation and monitoring modules.

This package provides:
- Model stability monitoring
- Drift detection
- Advanced evaluation metrics
- Dashboard-ready exports
"""

from .monitoring import (
    calculate_rolling_rmse,
    calculate_psi,
    detect_feature_drift,
    calculate_model_stability_index,
    create_monitoring_dashboard,
    create_all_monitoring_dashboards
)

__all__ = [
    'calculate_rolling_rmse',
    'calculate_psi',
    'detect_feature_drift',
    'calculate_model_stability_index',
    'create_monitoring_dashboard',
    'create_all_monitoring_dashboards'
]


