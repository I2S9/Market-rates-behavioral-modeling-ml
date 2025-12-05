# Financial Rates Forecasting & Behavioral Modeling

> A production-ready machine learning pipeline for forecasting market interest rates and modeling customer behavioral patterns in financial services. This project demonstrates ALM/Treasury modeling capabilities with statistical baselines, advanced ML models, and comprehensive monitoring dashboards.

## Context

This project addresses two critical challenges in financial services:

1. **Interest Rate Forecasting**: Predict future movements of key market rates (10-year Treasury, 30-year Treasury, Federal Funds Rate) to support Asset-Liability Management (ALM) and Treasury operations.

2. **Behavioral Modeling**: Classify customer events such as deposit outflows and early loan repayments to improve risk management and liquidity planning.

## Architecture

```
Market-rates-behavioral-modeling-ml/
├── data/
│   ├── raw/              # Raw CSV files
│   ├── processed/        # Cleaned and feature-engineered datasets
│   └── database.db       # SQLite database
├── src/
│   ├── data/             # Data ingestion and SQL pipeline
│   ├── features/         # Preprocessing and feature engineering
│   ├── models/           # Model training (baseline, ML, LSTM, behavioral)
│   └── evaluation/       # Stability monitoring and drift detection
├── reports/              # Model predictions, metrics, visualizations
└── dashboards/          # Power BI-ready monitoring data
```

## Pipeline

### 1. Data Ingestion
```bash
python src/data/generate_financial_data.py
python src/data/generate_behavioral_data.py
python src/data/sql_ingestion.py
```

### 2. Preprocessing & Feature Engineering
```bash
python src/features/preprocessing.py
python src/features/feature_engineering.py
```

**Features Generated**: 53 features including lags (1, 5, 10 days), rolling statistics (7, 30-day windows), volatility measures, and calendar features.

### 3. Model Training
```bash
python src/models/baseline_models.py      # ARIMA, Prophet
python src/models/ml_models.py            # Random Forest, XGBoost
python src/models/lstm_model.py           # LSTM
python src/models/behavioral_models.py    # Classification models
```

### 4. Monitoring
```bash
python src/evaluation/monitoring.py
```

## Results

### Forecasting Performance

**Model Comparison (Average RMSE across all series)**:

| Model | RMSE | Improvement vs Baseline |
|-------|------|------------------------|
| ARIMA | 0.1799 | Baseline |
| Prophet | 0.1581 | 12.1% |
| Random Forest | 0.0090 | 95.0% |
| **XGBoost** | **0.0070** | **96.1%** |

**Key Findings**:
- XGBoost achieves the best performance with 96% RMSE reduction vs baseline models
- Feature engineering (53 features) provides dramatic improvement (95% reduction from Prophet to RF)
- LSTM shows potential but requires more tuning (RMSE: 0.0536-0.0779)

**DGS10 Forecast Visualization**:
![DGS10 LSTM Forecast](reports/DGS10_lstm_forecast.png)

The LSTM captures general trends but shows higher variance compared to tree-based models. XGBoost provides more accurate predictions with RMSE of 0.0079.

### Behavioral Modeling Performance

**Classification Results**:

| Event Type | Best Model | AUC | Average Precision |
|------------|------------|-----|-------------------|
| Outflow Events | Random Forest | 0.6554 | 0.2519 |
| Early Repayment | Random Forest | 0.6907 | 0.2192 |

**ROC Curves - Outflow Events**:
![Outflow ROC Curves](reports/outflow_event_roc_curves.png)

Random Forest achieves the best discrimination for behavioral events. The models show reasonable performance given class imbalance (14.9% positive rate for outflows, 9.9% for early repayments).

**Note**: Performance on simulated data. Real-world implementations with domain-specific features typically achieve AUC > 0.75.

### Model Stability & Monitoring

**Stability Metrics (XGBoost models)**:

| Target | Stability Index | Mean Rolling RMSE | Features with Drift |
|--------|----------------|-------------------|---------------------|
| DGS10 | 0.6558 | 0.0071 | 25/53 (47%) |
| DFF | 0.7427 | 0.0084 | 25/53 (47%) |
| DGS30 | 0.6620 | 0.0063 | 25/53 (47%) |

**Key Insights**:
- DFF model shows best stability (0.7427) with consistent performance over time
- 47% of features show significant drift, primarily in temporal and volatility features
- Rolling RMSE remains stable, indicating no model degradation

## Dashboard

The project generates Power BI-ready monitoring dashboards:

**Files in `dashboards/`**:
- `*_monitoring.csv`: Daily predictions with rolling RMSE, stability index, temporal aggregations
- `*_drift_report.csv`: Feature drift detection (PSI scores)
- `*_monitoring_summary.json`: Aggregated metrics

**Key Columns for Power BI**:
- `date`, `actual`, `predicted`, `error`, `rolling_rmse`, `stability_index`
- `year`, `month`, `quarter` for temporal analysis

## How to Reproduce

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
git clone <repository-url>
cd Market-rates-behavioral-modeling-ml
pip install -r requirements.txt
```

### Complete Pipeline
```bash
# Step 1: Generate and ingest data
python src/data/generate_financial_data.py
python src/data/generate_behavioral_data.py
python src/data/sql_ingestion.py

# Step 2: Preprocess and engineer features
python src/features/preprocessing.py
python src/features/feature_engineering.py

# Step 3: Train models
python src/models/baseline_models.py
python src/models/ml_models.py
python src/models/lstm_model.py
python src/models/behavioral_models.py

# Step 4: Generate monitoring dashboards
python src/evaluation/monitoring.py
```

### Expected Outputs
- **Data**: `data/processed/rates_clean.csv`, `data/processed/rates_features.csv`
- **Predictions**: `reports/*_predictions.csv`
- **Metrics**: `reports/*_metrics.json`
- **Visualizations**: `reports/*_forecast.png`, `reports/*_roc_curves.png`
- **Dashboards**: `dashboards/*_monitoring.csv`

## Technical Stack

- **Data Processing**: pandas, numpy
- **Statistical Models**: statsmodels, pmdarima, prophet
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Deep Learning**: PyTorch
- **Database**: SQLite
- **Visualization**: matplotlib
- **Evaluation**: scipy

## Key Achievements

- **96% RMSE improvement** over baseline models through advanced feature engineering
- **Comprehensive model coverage** from statistical baselines to deep learning
- **Robust monitoring** with stability indices and drift detection
- **Production-ready outputs** with Power BI-compatible dashboards

The implementation follows ALM/Treasury best practices, emphasizing reproducibility, stability monitoring, and actionable insights for risk management.