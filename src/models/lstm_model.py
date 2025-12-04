"""
LSTM model for time series forecasting using PyTorch.

This module implements LSTM neural networks to capture complex
temporal dependencies in financial time series.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings
import json
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed. LSTM will be unavailable.")
    print("Install with: pip install torch")

# Scikit-learn for scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


class TimeSeriesDataset(Dataset):
    """Dataset for time series sequences."""
    
    def __init__(self, X, y):
        self.X = X  # Already sequences: (n_samples, sequence_length, n_features)
        self.y = y  # Already targets: (n_samples,)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.X[idx]),
            torch.FloatTensor([self.y[idx]])
        )


class LSTMModel(nn.Module):
    """Simple LSTM model with one LSTM layer and one Dense layer."""
    
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the last output
        out = self.fc(out)
        return out


def create_sequences(
    data: np.ndarray,
    sequence_length: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences from time series data.
    
    Args:
        data: 2D array (samples, features)
        sequence_length: Length of input sequences
        
    Returns:
        Tuple of (X sequences, y targets)
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length, 0])  # First column is target
    
    return np.array(X), np.array(y)


def prepare_lstm_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    sequence_length: int = 30
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler, pd.Index]:
    """
    Prepare data for LSTM training.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        feature_cols: List of feature columns to use
        sequence_length: Length of input sequences
        
    Returns:
        Tuple of (X_scaled, y_scaled, X_scaler, y_scaler, dates_index)
    """
    if feature_cols is None:
        # Use all numeric columns except target and other rates
        exclude = [target_col, 'DGS10', 'DFF', 'DGS30']
        feature_cols = [col for col in df.columns 
                       if col not in exclude 
                       and df[col].dtype in [np.float64, np.int64, float, int]]
    
    # Prepare features (include target as first column for sequences)
    data = df[[target_col] + feature_cols].copy()
    data = data.dropna()
    
    # Store dates for later
    dates_index = data.index
    
    # Scale data
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    # Scale features
    X_data = X_scaler.fit_transform(data[[target_col] + feature_cols].values)
    
    # Scale target separately for inverse transform
    y_data = y_scaler.fit_transform(data[[target_col]].values)
    
    # Replace first column (target) in X_data with scaled target
    X_data[:, 0] = y_data.flatten()
    
    # Create sequences
    X_seq, y_seq = create_sequences(X_data, sequence_length)
    
    # Adjust dates index for sequences (remove first sequence_length dates)
    dates_seq = dates_index[sequence_length:]
    
    return X_seq, y_seq, X_scaler, y_scaler, dates_seq


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    input_dim: int = 10,
    hidden_dim: int = 64,
    num_layers: int = 1,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: Optional[str] = None
) -> Tuple[LSTMModel, Dict]:
    """
    Train LSTM model.
    
    Args:
        X_train: Training sequences
        y_train: Training targets
        X_val: Validation sequences (optional)
        y_val: Validation targets (optional)
        input_dim: Input feature dimension
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        Tuple of (trained model, training history)
    """
    if not HAS_TORCH:
        return None, {'error': 'PyTorch not installed'}
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TimeSeriesDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = LSTMModel(input_dim, hidden_dim, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            model.train()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        else:
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}")
    
    return model, history


def predict_lstm(
    model: LSTMModel,
    X: np.ndarray,
    y_scaler: MinMaxScaler,
    device: Optional[str] = None
) -> np.ndarray:
    """
    Generate predictions with LSTM model.
    
    Args:
        model: Trained LSTM model
        X: Input sequences
        y_scaler: Scaler for inverse transform
        device: Device to use
        
    Returns:
        Array of predictions (inverse scaled)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(len(X)):
            x_seq = torch.FloatTensor(X[i:i+1]).to(device)
            pred = model(x_seq)
            predictions.append(pred.cpu().numpy()[0, 0])
    
    # Inverse transform
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = y_scaler.inverse_transform(predictions)
    
    return predictions.flatten()


def plot_forecast_vs_actual(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str,
    save_path: str
):
    """
    Plot forecast vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true.values, label='Actual', linewidth=2)
    plt.plot(y_true.index, y_pred, label='LSTM Forecast', linewidth=2, alpha=0.8)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Rate', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def train_lstm_models(
    features_path: str = "data/processed/rates_features.csv",
    target_columns: Optional[List[str]] = None,
    sequence_length: int = 30,
    test_size: float = 0.2,
    ml_metrics_path: str = "reports",
    output_dir: str = "reports"
) -> Dict:
    """
    Train LSTM models and compare with ML models.
    
    Args:
        features_path: Path to features CSV
        target_columns: List of target columns to forecast
        sequence_length: Length of input sequences (20-50)
        test_size: Proportion of data for testing
        ml_metrics_path: Path to ML metrics JSON files
        output_dir: Directory to save results
        
    Returns:
        Dictionary with model results
    """
    if not HAS_TORCH:
        print("PyTorch not available. Cannot train LSTM models.")
        return {}
    
    print("="*60)
    print("Training LSTM Models")
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
        print(f"Forecasting {target_col} with LSTM")
        print(f"{'='*60}")
        
        # Prepare data
        print(f"\nPreparing sequences (length={sequence_length})...")
        X_seq, y_seq, X_scaler, y_scaler, dates_seq = prepare_lstm_data(
            df, target_col, sequence_length=sequence_length
        )
        print(f"  Created {len(X_seq)} sequences")
        print(f"  Input dimension: {X_seq.shape[2]}")
        
        # Split data (temporal, no shuffle)
        split_idx = int(len(X_seq) * (1 - test_size))
        X_train = X_seq[:split_idx]
        y_train = y_seq[:split_idx]
        X_test = X_seq[split_idx:]
        y_test = y_seq[split_idx:]
        
        # Get test dates and actual values for plotting
        test_dates = dates_seq[split_idx:]
        y_test_actual = df[target_col].loc[test_dates]
        
        print(f"  Train: {len(X_train)} sequences, Test: {len(X_test)} sequences")
        
        # Train model
        print(f"\nTraining LSTM model...")
        input_dim = X_seq.shape[2]
        model, history = train_lstm(
            X_train, y_train,
            input_dim=input_dim,
            hidden_dim=64,
            num_layers=1,
            epochs=50,
            batch_size=32
        )
        
        if model is None:
            print(f"  Error: {history.get('error', 'Unknown error')}")
            continue
        
        # Predictions
        print(f"\nGenerating predictions...")
        y_pred_scaled = predict_lstm(model, X_test, y_scaler)
        
        # Calculate metrics
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test_actual.values, y_pred_scaled)),
            'MAE': mean_absolute_error(y_test_actual.values, y_pred_scaled)
        }
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
        
        # Compare with ML models
        ml_file = Path(ml_metrics_path) / f"{target_col}_ml_metrics.json"
        if ml_file.exists():
            with open(ml_file, 'r') as f:
                ml_data = json.load(f)
                best_ml = min(ml_data.values(), key=lambda x: x['RMSE'])
                print(f"\n  ML best (XGBoost): RMSE={best_ml['RMSE']:.4f}")
                improvement = ((best_ml['RMSE'] - metrics['RMSE']) / best_ml['RMSE']) * 100
                print(f"  LSTM vs ML: {improvement:+.2f}% RMSE")
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'date': test_dates,
            'actual': y_test_actual.values,
            'lstm_pred': y_pred_scaled
        }, index=test_dates)
        
        pred_file = output_path / f"{target_col}_lstm_predictions.csv"
        predictions_df.to_csv(pred_file)
        print(f"\n  Saved predictions to {pred_file}")
        
        # Save metrics
        metrics_file = output_path / f"{target_col}_lstm_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({'LSTM': metrics}, f, indent=2)
        print(f"  Saved metrics to {metrics_file}")
        
        # Plot forecast vs actual
        plot_file = output_path / f"{target_col}_lstm_forecast.png"
        plot_forecast_vs_actual(
            y_test_actual,
            y_pred_scaled,
            f"{target_col} - LSTM Forecast vs Actual",
            str(plot_file)
        )
        print(f"  Saved plot to {plot_file}")
        
        results[target_col] = {
            'model': model,
            'predictions': y_pred_scaled,
            'metrics': metrics,
            'history': history
        }
    
    print(f"\n{'='*60}")
    print("LSTM Training Complete")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    results = train_lstm_models(
        target_columns=['DGS10', 'DFF', 'DGS30'],
        sequence_length=30,
        test_size=0.2
    )
    
    print("\nSummary of Results:")
    for target, data in results.items():
        metrics = data['metrics']
        print(f"\n{target}:")
        print(f"  LSTM: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")

