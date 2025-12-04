"""
Feature engineering and preprocessing modules.

This package provides utilities for:
- Time series preprocessing and cleaning
- Feature engineering pipelines
"""

from .preprocessing import (
    load_rates_from_db,
    uniformize_frequency,
    handle_missing_values,
    test_stationarity,
    normalize_series,
    preprocess_rates
)

__all__ = [
    'load_rates_from_db',
    'uniformize_frequency',
    'handle_missing_values',
    'test_stationarity',
    'normalize_series',
    'preprocess_rates'
]

