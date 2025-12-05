"""
Download financial time series from FRED (Federal Reserve Economic Data).

This module downloads publicly available interest rate data from FRED API.
Requires a free FRED API key (available at https://fred.stlouisfed.org/docs/api/api_key.html).
"""

import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path
import requests


def download_fred_series(series_id: str, api_key: str = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Download a time series from FRED API.
    
    Args:
        series_id: FRED series identifier (e.g., 'DGS10' for 10-Year Treasury Rate)
        api_key: FRED API key (optional, but recommended for higher rate limits)
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with date index and series values
    """
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    
    params = {
        'series_id': series_id,
        'file_type': 'json',
        'sort_order': 'asc'
    }
    
    if api_key:
        params['api_key'] = api_key
    
    if start_date:
        params['observation_start'] = start_date
    else:
        params['observation_start'] = (datetime.now() - timedelta(days=365*6)).strftime('%Y-%m-%d')
    
    if end_date:
        params['observation_end'] = end_date
    else:
        params['observation_end'] = datetime.now().strftime('%Y-%m-%d')
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'observations' not in data:
            raise ValueError(f"No observations found for series {series_id}")
        
        df = pd.DataFrame(data['observations'])
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df[['date', 'value']].rename(columns={'value': series_id})
        df = df.set_index('date')
        df = df.dropna()
        
        return df
    
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to download data from FRED: {e}")


def download_multiple_series(series_dict: dict, api_key: str = None, output_dir: str = "data/raw") -> None:
    """
    Download multiple FRED series and save to CSV files.
    
    Args:
        series_dict: Dictionary mapping series_id to description
        api_key: FRED API key (optional)
        output_dir: Directory to save CSV files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    start_date = (datetime.now() - timedelta(days=365*6)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    for series_id, description in series_dict.items():
        print(f"Downloading {series_id}: {description}...")
        try:
            df = download_fred_series(series_id, api_key, start_date, end_date)
            
            if len(df) > 0:
                filename = output_path / f"{series_id}.csv"
                df.to_csv(filename)
                print(f"  Saved {len(df)} observations to {filename}")
            else:
                print(f"  Warning: No data retrieved for {series_id}")
        
        except Exception as e:
            print(f"  Error downloading {series_id}: {e}")


if __name__ == "__main__":
    # FRED series identifiers for interest rates
    # DGS10: 10-Year Treasury Constant Maturity Rate
    # DFF: Federal Funds Effective Rate
    # DGS30: 30-Year Treasury Constant Maturity Rate
    # DEXUSEU: U.S. / Euro Foreign Exchange Rate
    
    series_to_download = {
        'DGS10': '10-Year Treasury Constant Maturity Rate',
        'DFF': 'Federal Funds Effective Rate',
        'DGS30': '30-Year Treasury Constant Maturity Rate'
    }
    
    # API key is optional but recommended
    # Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html
    api_key = os.getenv('FRED_API_KEY', None)
    
    if not api_key:
        print("Warning: No FRED_API_KEY found. Using unauthenticated requests (lower rate limits).")
        print("Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
    
    download_multiple_series(series_to_download, api_key=api_key)


