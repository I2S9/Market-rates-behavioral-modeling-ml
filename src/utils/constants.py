"""Project-wide constants for reproducibility."""

# Random seed for reproducibility
RANDOM_SEED = 42

# Default target columns
DEFAULT_TARGET_COLUMNS = ['DGS10', 'DFF', 'DGS30']

# Default behavioral targets
DEFAULT_BEHAVIORAL_TARGETS = ['outflow_event', 'early_repayment_event']

# Default paths
DEFAULT_DATA_PATH = "data/processed/rates_clean.csv"
DEFAULT_FEATURES_PATH = "data/processed/rates_features.csv"
DEFAULT_BEHAVIORAL_PATH = "data/raw/behavioral_data.csv"
DEFAULT_OUTPUT_DIR = "reports"
DEFAULT_DASHBOARD_DIR = "dashboards"

# Model parameters
DEFAULT_N_ESTIMATORS = 300
DEFAULT_TEST_SIZE = 0.2
DEFAULT_WINDOW_SIZE = 30

