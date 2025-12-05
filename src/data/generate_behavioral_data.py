"""
Generate simulated behavioral dataset for customer deposit outflows and early repayments.

This module creates realistic synthetic data with features relevant to behavioral modeling:
- Balance volatility
- Seasonality patterns
- Macro indicators
- Customer characteristics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def generate_behavioral_dataset(n_samples: int = 10000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic behavioral dataset for customer modeling.
    
    Args:
        n_samples: Number of customer records to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with features and target variables
    """
    np.random.seed(random_seed)
    
    # Generate date range (last 3 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    
    # Customer IDs
    customer_ids = [f"CUST_{i:06d}" for i in range(n_samples)]
    
    # Customer segment (affects behavior)
    segments = np.random.choice(['Retail', 'Corporate', 'Institutional'], 
                                size=n_samples, 
                                p=[0.6, 0.3, 0.1])
    
    # Account age in months
    account_age_months = np.random.gamma(shape=2, scale=12, size=n_samples).astype(int)
    account_age_months = np.clip(account_age_months, 1, 120)
    
    # Base balance (log-normal distribution)
    base_balance = np.random.lognormal(mean=10, sigma=1.5, size=n_samples)
    base_balance = np.clip(base_balance, 1000, 10000000)
    
    # Balance volatility (monthly coefficient of variation)
    balance_volatility = np.random.gamma(shape=1.5, scale=0.05, size=n_samples)
    balance_volatility = np.clip(balance_volatility, 0.01, 0.5)
    
    # Seasonal component (stronger in Q4 for retail)
    month = pd.to_datetime(dates).month
    is_q4 = (month >= 10) & (month <= 12)
    seasonal_factor = np.where(is_q4, 1.2, 1.0)
    seasonal_factor = np.where((segments == 'Retail') & is_q4, 1.4, seasonal_factor)
    
    # Macro indicators (simulated)
    # Interest rate level (affects outflow probability)
    interest_rate_level = 2.5 + np.random.normal(0, 0.5, size=n_samples)
    interest_rate_level = np.clip(interest_rate_level, 0.5, 5.0)
    
    # Economic stress indicator (0-1 scale)
    economic_stress = np.random.beta(a=2, b=5, size=n_samples)
    
    # Market volatility (VIX-like proxy)
    market_volatility = np.random.gamma(shape=2, scale=5, size=n_samples)
    market_volatility = np.clip(market_volatility, 5, 50)
    
    # Customer relationship features
    num_products = np.random.poisson(lam=2.5, size=n_samples)
    num_products = np.clip(num_products, 1, 8)
    
    # Transaction frequency (monthly)
    transaction_frequency = np.random.gamma(shape=3, scale=2, size=n_samples)
    transaction_frequency = np.clip(transaction_frequency, 0.5, 30)
    
    # Credit score (for early repayment modeling)
    credit_score = np.random.normal(700, 50, size=n_samples)
    credit_score = np.clip(credit_score, 500, 850)
    
    # Generate target: Outflow probability (0-1)
    # Higher volatility, higher interest rates, economic stress increase outflow
    outflow_logit = (
        -2.5
        + 1.5 * balance_volatility
        + 0.3 * (interest_rate_level - 2.5)
        + 2.0 * economic_stress
        + 0.5 * (segments == 'Retail').astype(int)
        - 0.2 * np.log(base_balance) / 10
        - 0.1 * account_age_months / 12
        + np.random.normal(0, 0.5, size=n_samples)
    )
    outflow_probability = 1 / (1 + np.exp(-outflow_logit))
    outflow_probability = np.clip(outflow_probability, 0, 1)
    
    # Binary outflow event (using probability)
    outflow_event = (np.random.random(n_samples) < outflow_probability).astype(int)
    
    # Early repayment event (for loans)
    # Higher credit score, lower interest rates, higher stress increase early repayment
    early_repayment_logit = (
        -3.0
        + 0.01 * (credit_score - 700)
        - 0.4 * (interest_rate_level - 2.5)
        + 1.5 * economic_stress
        + 0.3 * (segments == 'Corporate').astype(int)
        + np.random.normal(0, 0.6, size=n_samples)
    )
    early_repayment_probability = 1 / (1 + np.exp(-early_repayment_logit))
    early_repayment_probability = np.clip(early_repayment_probability, 0, 1)
    early_repayment_event = (np.random.random(n_samples) < early_repayment_probability).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'customer_id': customer_ids,
        'customer_segment': segments,
        'account_age_months': account_age_months,
        'base_balance': base_balance,
        'balance_volatility': balance_volatility,
        'seasonal_factor': seasonal_factor,
        'interest_rate_level': interest_rate_level,
        'economic_stress': economic_stress,
        'market_volatility': market_volatility,
        'num_products': num_products,
        'transaction_frequency': transaction_frequency,
        'credit_score': credit_score,
        'outflow_probability': outflow_probability,
        'outflow_event': outflow_event,
        'early_repayment_probability': early_repayment_probability,
        'early_repayment_event': early_repayment_event
    })
    
    return df


if __name__ == "__main__":
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating behavioral dataset...")
    df = generate_behavioral_dataset(n_samples=12000, random_seed=42)
    
    output_file = output_dir / "behavioral_data.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Generated {len(df)} records")
    print(f"Saved to {output_file}")
    print(f"\nDataset summary:")
    print(f"  - Outflow events: {df['outflow_event'].sum()} ({df['outflow_event'].mean()*100:.1f}%)")
    print(f"  - Early repayment events: {df['early_repayment_event'].sum()} ({df['early_repayment_event'].mean()*100:.1f}%)")
    print(f"\nFeatures:")
    print(df.describe())


