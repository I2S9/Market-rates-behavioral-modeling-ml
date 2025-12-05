"""
Generate realistic simulated financial time series data.

This module creates synthetic interest rate data based on historical patterns
and statistical properties of real financial series. Useful when API access
is not available or for testing purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def generate_interest_rate_series(
    name: str,
    start_date: datetime,
    end_date: datetime,
    initial_rate: float,
    mean_reversion_level: float,
    volatility: float,
    random_seed: int = None
) -> pd.DataFrame:
    """
    Generate realistic interest rate time series using mean-reverting process.
    
    Args:
        name: Series name/identifier
        start_date: Start date for the series
        end_date: End date for the series
        initial_rate: Starting interest rate value
        mean_reversion_level: Long-term mean rate
        volatility: Annual volatility of rate changes
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with date index and rate values
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate daily dates
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Convert annual volatility to daily
    daily_vol = volatility / np.sqrt(252)
    
    # Mean reversion speed
    mean_reversion_speed = 0.1
    
    # Initialize rate series
    rates = np.zeros(n_days)
    rates[0] = initial_rate
    
    # Generate random shocks with some autocorrelation
    shocks = np.random.normal(0, daily_vol, n_days)
    
    # Add some trend and cyclical components
    trend = np.linspace(0, (mean_reversion_level - initial_rate) * 0.3, n_days)
    
    # Business cycle component (4-7 year cycles)
    cycle_period = 252 * 5  # 5 years in trading days
    cycle = 0.3 * np.sin(2 * np.pi * np.arange(n_days) / cycle_period)
    
    # Seasonal component (weaker for rates)
    seasonal = 0.05 * np.sin(2 * np.pi * pd.to_datetime(dates).dayofyear / 365.25)
    
    # Generate rates using mean-reverting process
    for i in range(1, n_days):
        # Mean-reverting component
        mean_revert = mean_reversion_speed * (mean_reversion_level - rates[i-1])
        
        # Random shock
        random_shock = shocks[i]
        
        # Combine all components
        rates[i] = rates[i-1] + mean_revert + random_shock + trend[i] - trend[i-1] + cycle[i] - cycle[i-1] + seasonal[i] - seasonal[i-1]
        
        # Ensure rates stay positive and reasonable
        rates[i] = np.clip(rates[i], 0.1, 15.0)
    
    # Convert to DataFrame
    df = pd.DataFrame({name: rates}, index=dates)
    
    # Resample to business days only (remove weekends)
    df = df[df.index.weekday < 5]
    
    return df


def generate_multiple_series(output_dir: str = "data/raw") -> None:
    """
    Generate multiple financial time series and save to CSV files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate 6+ years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*6 + 30)
    
    series_configs = [
        {
            'name': 'DGS10',
            'description': '10-Year Treasury Constant Maturity Rate',
            'initial_rate': 3.5,
            'mean_reversion': 3.2,
            'volatility': 0.8,
            'seed': 42
        },
        {
            'name': 'DFF',
            'description': 'Federal Funds Effective Rate',
            'initial_rate': 4.5,
            'mean_reversion': 4.0,
            'volatility': 1.2,
            'seed': 43
        },
        {
            'name': 'DGS30',
            'description': '30-Year Treasury Constant Maturity Rate',
            'initial_rate': 4.0,
            'mean_reversion': 3.8,
            'volatility': 0.6,
            'seed': 44
        }
    ]
    
    for config in series_configs:
        print(f"Generating {config['name']}: {config['description']}...")
        
        df = generate_interest_rate_series(
            name=config['name'],
            start_date=start_date,
            end_date=end_date,
            initial_rate=config['initial_rate'],
            mean_reversion_level=config['mean_reversion'],
            volatility=config['volatility'],
            random_seed=config['seed']
        )
        
        filename = output_path / f"{config['name']}.csv"
        df.to_csv(filename)
        print(f"  Saved {len(df)} observations to {filename}")
        print(f"  Rate range: {df[config['name']].min():.2f}% - {df[config['name']].max():.2f}%")


if __name__ == "__main__":
    generate_multiple_series()


