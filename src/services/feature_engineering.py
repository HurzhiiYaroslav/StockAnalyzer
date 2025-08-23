import pandas as pd
import numpy as np

from src.services.fetch_data import fetch_single_ticker_history
from src.services import metadata_service
from src.services.technical_indicators import calculate_historical_features_for_ticker
from src.services.fundamental_data_processing import get_fundamental_features_for_ticker

def generate_features_for_ticker(
    ticker_symbol: str, 
    start_date_str: str, 
    end_date_str: str
) -> tuple[pd.DataFrame, list]:
    """Orchestrates the generation of a complete feature set for a single ticker."""
    print(f"\n=== Generating features for ticker {ticker_symbol} ===")
    
    historical_data = fetch_single_ticker_history(
        ticker_symbol=ticker_symbol,
        start_date=start_date_str,
        end_date=end_date_str,
        interval="1d"
    )
    if historical_data.empty:
        print(f"  Error: Failed to fetch historical price data for {ticker_symbol}.")
        return pd.DataFrame(), []

    technical_features = calculate_historical_features_for_ticker(
        df_ohlcv=historical_data,
        ticker_symbol=ticker_symbol
    )
    if technical_features.empty:
        print(f"  Warning: Could not calculate technical features for {ticker_symbol}. Using basic OHLCV.")
        technical_features = historical_data.copy()
        technical_features['ticker'] = ticker_symbol

    fundamental_features, fundamental_metrics_list = get_fundamental_features_for_ticker(
        ticker_symbol=ticker_symbol,
        start_date_str=start_date_str,
        historical_data=historical_data
    )
    if fundamental_features.empty:
        print(f"  Warning: Could not calculate fundamental features for {ticker_symbol}.")
        fundamental_features = pd.DataFrame(index=historical_data.index)
        fundamental_metrics_list = []
    
    fundamental_features.index = pd.to_datetime(fundamental_features.index).normalize()
    technical_features.index = pd.to_datetime(technical_features.index).normalize()

    combined_features = pd.merge(
        technical_features, 
        fundamental_features.drop(columns=['ticker'], errors='ignore'), 
        left_index=True,
        right_index=True,
        how='left'
    )
    if combined_features.empty:
        print(f"  Error: Failed to combine technical and fundamental data for {ticker_symbol}.")
        return pd.DataFrame(), []

    print(f"  Getting metadata for {ticker_symbol}...")
    ticker_metadata = metadata_service.get_for_ticker(ticker_symbol)
    sector = ticker_metadata.get('sector', 'Unknown')
    combined_features['sector'] = sector
    
    if 'close' in combined_features.columns and 'ttm_eps' in combined_features.columns:
        combined_features['P_E_Ratio'] = np.where(
            (combined_features['ttm_eps'].notna()) & (combined_features['ttm_eps'] != 0),
            combined_features['close'] / combined_features['ttm_eps'], 
            np.nan
        )
        if 'P_E_Ratio' not in fundamental_metrics_list:
             fundamental_metrics_list.append('P_E_Ratio')

    fund_cols_to_fill = [col for col in fundamental_metrics_list if col in combined_features.columns]
    
    if fund_cols_to_fill:
        combined_features[fund_cols_to_fill] = combined_features[fund_cols_to_fill].ffill()

    print(f"  Successfully generated feature set for {ticker_symbol} ({len(combined_features)} rows).")
    
    return combined_features, fundamental_metrics_list