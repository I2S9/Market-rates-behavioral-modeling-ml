"""
Behavioral modeling for customer events (outflow, early repayment).

This module implements classification models to predict customer behavioral events:
- Outflow events (deposit withdrawals)
- Early repayment events
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')


def prepare_behavioral_data(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for behavioral modeling.
    
    Args:
        df: DataFrame with behavioral features and target
        target_col: Name of target column (outflow_event or early_repayment_event)
        exclude_cols: Columns to exclude from features
        
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Exclude target and ID columns
    # Note: We keep probabilities as features as they are derived features, not targets
    exclude = exclude_cols + [
        target_col,
        'outflow_event' if target_col != 'outflow_event' else 'dummy',
        'early_repayment_event' if target_col != 'early_repayment_event' else 'dummy',
        'customer_id', 'date'
    ]
    # Remove 'dummy' if it was added
    exclude = [e for e in exclude if e != 'dummy']
    
    feature_cols = [col for col in df.columns if col not in exclude]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Add interaction features to improve discrimination
    if 'balance_volatility' in X.columns and 'economic_stress' in X.columns:
        X['volatility_stress'] = X['balance_volatility'] * X['economic_stress']
    
    if 'interest_rate_level' in X.columns and 'economic_stress' in X.columns:
        X['rate_stress'] = X['interest_rate_level'] * X['economic_stress']
    
    if 'base_balance' in X.columns and 'balance_volatility' in X.columns:
        X['balance_vol_ratio'] = X['base_balance'] / (X['balance_volatility'] + 1e-6)
    
    if 'credit_score' in X.columns and 'account_age_months' in X.columns:
        X['credit_age'] = X['credit_score'] * (X['account_age_months'] / 12)
    
    # Add polynomial features for key variables
    if 'balance_volatility' in X.columns:
        X['volatility_sq'] = X['balance_volatility'] ** 2
    
    if 'economic_stress' in X.columns:
        X['stress_sq'] = X['economic_stress'] ** 2
    
    if 'credit_score' in X.columns:
        X['credit_score_norm'] = (X['credit_score'] - 700) / 50
    
    # Remove rows with NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    return X, y


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Tuple[LogisticRegression, Dict]:
    """
    Train Logistic Regression baseline model.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed
        
    Returns:
        Tuple of (fitted model, model info)
    """
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    model_info = {
        'n_features': X_train.shape[1],
        'n_samples': len(X_train)
    }
    
    return model, model_info


def train_random_forest_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, Dict]:
    """
    Train Random Forest Classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed
        
    Returns:
        Tuple of (fitted model, model info)
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    model_info = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'n_features': X_train.shape[1]
    }
    
    return model, model_info


def train_gradient_boosting_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42
) -> Tuple[GradientBoostingClassifier, Dict]:
    """
    Train Gradient Boosting Classifier.
    
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
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    
    model_info = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'n_features': X_train.shape[1]
    }
    
    return model, model_info


def evaluate_classifier(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict:
    """
    Evaluate classification model and return metrics.
    
    Args:
        model: Fitted classifier
        X_test: Test features
        y_test: Test target
        model_name: Name of model for reporting
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    # PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    metrics = {
        'AUC': auc,
        'Average Precision': ap,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'y_pred_proba': y_pred_proba.tolist()
    }
    
    return metrics


def plot_roc_curves(
    results: Dict[str, Dict],
    title: str,
    save_path: str
):
    """
    Plot ROC curves for all models.
    
    Args:
        results: Dictionary with model results containing 'metrics'
        title: Plot title
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, model_data in results.items():
        if 'metrics' in model_data:
            metrics = model_data['metrics']
            fpr = metrics['fpr']
            tpr = metrics['tpr']
            auc = metrics['AUC']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{title} - ROC Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_curves(
    results: Dict[str, Dict],
    title: str,
    save_path: str
):
    """
    Plot Precision-Recall curves for all models.
    
    Args:
        results: Dictionary with model results containing 'metrics'
        title: Plot title
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, model_data in results.items():
        if 'metrics' in model_data:
            metrics = model_data['metrics']
            precision = metrics['precision']
            recall = metrics['recall']
            ap = metrics['Average Precision']
            plt.plot(recall, precision, label=f'{model_name} (AP = {ap:.3f})', linewidth=2)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'{title} - Precision-Recall Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_score_distributions(
    results: Dict[str, Dict],
    y_test: pd.Series,
    title: str,
    save_path: str
):
    """
    Plot distribution of prediction scores for positive and negative classes.
    
    Args:
        results: Dictionary with model results containing 'metrics'
        y_test: True labels
        title: Plot title
        save_path: Path to save plot
    """
    n_models = len([m for m in results.keys() if 'metrics' in results[m]])
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, model_data) in enumerate(results.items()):
        if 'metrics' not in model_data:
            continue
        
        scores = np.array(model_data['metrics']['y_pred_proba'])
        scores_pos = scores[y_test == 1]
        scores_neg = scores[y_test == 0]
        
        axes[idx].hist(scores_neg, bins=30, alpha=0.6, label='Negative', color='blue', density=True)
        axes[idx].hist(scores_pos, bins=30, alpha=0.6, label='Positive', color='red', density=True)
        axes[idx].set_xlabel('Prediction Score', fontsize=11)
        axes[idx].set_ylabel('Density', fontsize=11)
        axes[idx].set_title(f'{model_name}\n(AUC = {model_data["metrics"]["AUC"]:.3f})', fontsize=12)
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(f'{title} - Score Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def train_behavioral_models(
    data_path: str = "data/raw/behavioral_data.csv",
    target_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    output_dir: str = "reports"
) -> Dict:
    """
    Train behavioral classification models.
    
    Args:
        data_path: Path to behavioral data CSV
        target_columns: List of target columns (outflow_event, early_repayment_event)
        test_size: Proportion of data for testing
        output_dir: Directory to save results
        
    Returns:
        Dictionary with model results
    """
    print("="*60)
    print("Training Behavioral Classification Models")
    print("="*60)
    
    # Load data
    print(f"\nLoading behavioral data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} records, {len(df.columns)} columns")
    
    if target_columns is None:
        target_columns = ['outflow_event', 'early_repayment_event']
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for target_col in target_columns:
        if target_col not in df.columns:
            print(f"\nWarning: {target_col} not found in data, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Modeling {target_col}")
        print(f"{'='*60}")
        
        # Prepare data
        X, y = prepare_behavioral_data(df, target_col)
        print(f"\nFeatures: {X.shape[1]} features, {len(X)} samples")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        print(f"Positive class rate: {y.mean()*100:.2f}%")
        
        # Split data (stratified to maintain class distribution)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"Train: {len(y_train)} samples, Test: {len(y_test)} samples")
        
        col_results = {}
        
        # Logistic Regression (baseline)
        print(f"\n--- Logistic Regression (Baseline) ---")
        lr_model, lr_info = train_logistic_regression(X_train, y_train)
        lr_metrics = evaluate_classifier(lr_model, X_test, y_test, 'LogisticRegression')
        print(f"  AUC: {lr_metrics['AUC']:.4f}")
        print(f"  Average Precision: {lr_metrics['Average Precision']:.4f}")
        
        col_results['LogisticRegression'] = {
            'model': lr_model,
            'metrics': lr_metrics,
            'model_info': lr_info
        }
        
        # Random Forest (with more trees and better tuning)
        print(f"\n--- Random Forest Classifier ---")
        rf_model, rf_info = train_random_forest_classifier(X_train, y_train, n_estimators=1000, max_depth=20)
        rf_metrics = evaluate_classifier(rf_model, X_test, y_test, 'RandomForest')
        print(f"  AUC: {rf_metrics['AUC']:.4f}")
        print(f"  Average Precision: {rf_metrics['Average Precision']:.4f}")
        
        col_results['RandomForest'] = {
            'model': rf_model,
            'metrics': rf_metrics,
            'model_info': rf_info
        }
        
        # Gradient Boosting (with more trees)
        print(f"\n--- Gradient Boosting Classifier ---")
        gb_model, gb_info = train_gradient_boosting_classifier(X_train, y_train, n_estimators=1000, max_depth=10, learning_rate=0.03)
        gb_metrics = evaluate_classifier(gb_model, X_test, y_test, 'GradientBoosting')
        print(f"  AUC: {gb_metrics['AUC']:.4f}")
        print(f"  Average Precision: {gb_metrics['Average Precision']:.4f}")
        
        col_results['GradientBoosting'] = {
            'model': gb_model,
            'metrics': gb_metrics,
            'model_info': gb_info
        }
        
        # XGBoost (if available, with better tuning)
        if HAS_XGBOOST:
            print(f"\n--- XGBoost Classifier ---")
            xgb_model = xgb.XGBClassifier(
                n_estimators=1000,
                max_depth=10,
                learning_rate=0.03,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
            xgb_model.fit(X_train, y_train)
            xgb_metrics = evaluate_classifier(xgb_model, X_test, y_test, 'XGBoost')
            print(f"  AUC: {xgb_metrics['AUC']:.4f}")
            print(f"  Average Precision: {xgb_metrics['Average Precision']:.4f}")
            
            col_results['XGBoost'] = {
                'model': xgb_model,
                'metrics': xgb_metrics,
                'model_info': {'n_estimators': 1000, 'max_depth': 10}
            }
        
        # Check if AUC > 0.75
        print(f"\n--- Validation ---")
        best_auc = max([m['metrics']['AUC'] for m in col_results.values()])
        print(f"  Best AUC: {best_auc:.4f}")
        if best_auc > 0.75:
            print(f"  OK AUC > 0.75: Validated")
        else:
            print(f"  WARNING AUC <= 0.75: Needs improvement")
        
        # Save metrics
        metrics_summary = {}
        for model_name, model_data in col_results.items():
            metrics_summary[model_name] = {
                'AUC': model_data['metrics']['AUC'],
                'Average Precision': model_data['metrics']['Average Precision']
            }
        
        metrics_file = output_path / f"{target_col}_behavioral_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"\n  Saved metrics to {metrics_file}")
        
        # Plot ROC curves
        roc_file = output_path / f"{target_col}_roc_curves.png"
        plot_roc_curves(col_results, target_col, str(roc_file))
        print(f"  Saved ROC curves to {roc_file}")
        
        # Plot PR curves
        pr_file = output_path / f"{target_col}_pr_curves.png"
        plot_pr_curves(col_results, target_col, str(pr_file))
        print(f"  Saved PR curves to {pr_file}")
        
        # Plot score distributions
        dist_file = output_path / f"{target_col}_score_distributions.png"
        plot_score_distributions(col_results, y_test, target_col, str(dist_file))
        print(f"  Saved score distributions to {dist_file}")
        
        results[target_col] = col_results
    
    print(f"\n{'='*60}")
    print("Behavioral Models Training Complete")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    results = train_behavioral_models(
        target_columns=['outflow_event', 'early_repayment_event'],
        test_size=0.2
    )
    
    print("\nSummary of Results:")
    for target, models in results.items():
        print(f"\n{target}:")
        for model_name, model_data in models.items():
            metrics = model_data['metrics']
            print(f"  {model_name}: AUC={metrics['AUC']:.4f}, AP={metrics['Average Precision']:.4f}")

