import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from src import config
from src.services.fetch_data import get_fundamental_data, fetch_single_ticker_history
from src.services.technical_indicators import calculate_historical_features_for_ticker

def calculate_ttm(series, window=4, min_periods=4):
    if series is None or not isinstance(series, pd.Series) or series.empty:
        return pd.Series(dtype='float64', index=series.index if isinstance(series, pd.Series) else None)
    series_numeric = pd.to_numeric(series, errors='coerce')
    if series_numeric.isnull().all():
        return pd.Series(dtype='float64', index=series.index)
    series_sorted = series_numeric.sort_index()
    return series_sorted.rolling(window=window, min_periods=min_periods).sum()

def get_fundamental_features_for_ticker(ticker_symbol, start_date_str):
    print(f"\n--- Processing fundamental data for ticker: {ticker_symbol} ---")

    file_path = os.path.join(config.RAW_DATA_DIR, f"{ticker_symbol}.json")
    if not os.path.exists(file_path):
        print(f"  Data file not found, downloading fresh data...")
        if not get_fundamental_data(ticker_symbol):
            print(f"  Failed to get fundamental data for {ticker_symbol}")
            return pd.DataFrame()

    end_date = datetime.now().strftime('%Y-%m-%d')
    historical_data = fetch_single_ticker_history(
        ticker_symbol=ticker_symbol,
        start_date=start_date_str,
        end_date=end_date,
        interval="1d"
    )
    
    if historical_data.empty:
        print(f"  Unable to retrieve historical data for {ticker_symbol}")
        return pd.DataFrame()

    latest_file_name = f"{ticker_symbol}.json"
    file_path = os.path.join(config.RAW_DATA_DIR, latest_file_name)
    print(f"  Using file: {latest_file_name}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            data = json.loads(content) if content.strip() else {}
    except Exception as e:
        print(f"      Error reading/parsing JSON file {file_path}: {e}")
        data = {}
        
    quarterly_data_raw = data.get("financial_data", {}).get("quarterly", {})
    processed_quarters = {}
    for metric, quarterly_values in quarterly_data_raw.items():
        if isinstance(quarterly_values, dict):
            clean_metric_name = metric.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_per_')
            for quarter_str, value in quarterly_values.items():
                try:
                    year = int(quarter_str[:4])
                    q_num = int(quarter_str[5:])
                    month = q_num * 3
                    quarter_end_date = pd.Timestamp(datetime(year, month, 1) + pd.offsets.MonthEnd(0))
                    if quarter_end_date not in processed_quarters:
                        processed_quarters[quarter_end_date] = {}
                    processed_quarters[quarter_end_date][clean_metric_name] = value
                except (ValueError, TypeError):
                    continue

    q_df = pd.DataFrame.from_dict(processed_quarters, orient='index').sort_index()
    for col in q_df.columns: q_df[col] = pd.to_numeric(q_df[col], errors='coerce')

    daily_date_index = pd.date_range(start=start_date_str, end=datetime.now().strftime('%Y-%m-%d'), freq='B')
    daily_date_index.name = 'date'
    
    daily_fund_df = pd.DataFrame(index=daily_date_index)

    def find_col_name(df_cols, candidates):
        return next((c for c in candidates if c in df_cols), None)

    col_revenue = find_col_name(q_df.columns, ['total_revenue', 'revenue'])
    col_net_income = find_col_name(q_df.columns, ['net_income_common_stockholders', 'netincomeavailabletocommonstockholders', 'net_income_continuous_operations', 'net_income', 'netincome'])
    col_eps = find_col_name(q_df.columns, ['eps_diluted', 'eps_basic', 'diluted_eps_excluding_extraordinary_items'])
    
    if col_eps:
        daily_fund_df['ttm_eps'] = calculate_ttm(q_df.get(col_eps)).reindex(daily_date_index, method='ffill')

    if 'ttm_eps' in daily_fund_df and 'close' in historical_data.columns:
        historical_data.index = pd.to_datetime(historical_data.index).normalize()
        aligned_close = historical_data['close'].reindex(daily_date_index, method='ffill')
        pe_ratio = np.where(daily_fund_df['ttm_eps'].ne(0) & daily_fund_df['ttm_eps'].notna(), aligned_close / daily_fund_df['ttm_eps'], np.nan)
        daily_fund_df['P_E_Ratio'] = pd.Series(pe_ratio, index=daily_date_index).replace(0, np.nan)
    
    all_q_cols = list(q_df.columns)
    for col in all_q_cols:
        daily_fund_df[col] = q_df[col].reindex(daily_date_index, method='ffill')
    
    result_df = daily_fund_df.bfill().ffill()

    if result_df.empty or result_df.isnull().all().all():
        print(f"      Could not calculate significant fundamental metrics for {ticker_symbol}.")
    else:
        print(f"      Fundamental metrics for {ticker_symbol} calculated.")
        
    return result_df

def calculate_sector_z_scores(df: pd.DataFrame, metrics_to_normalize: list) -> pd.DataFrame:
    if 'sector' not in df.columns:
        print("  Warning: 'sector' column not found for Z-score calculation.")
        return df

    df_z = df.copy()
    
    original_index = df_z.index
    
    if 'date' not in df_z.index.names and 'date' in df_z.columns:
        df_z = df_z.set_index('date', append=True)
    
    print(f"  Calculating Z-scores for metrics: {metrics_to_normalize}")
    
    for metric in metrics_to_normalize:
        if metric in df_z.columns:
            grouped = df_z.groupby(['sector', 'date'])[metric]
            
            sector_mean = grouped.transform('mean')
            sector_std = grouped.transform('std')
            
            epsilon = 1e-6
            z_score_col_name = f"{metric}_z_score"
            df_z[z_score_col_name] = (df_z[metric] - sector_mean) / (sector_std + epsilon)
            df_z[z_score_col_name] = df_z[z_score_col_name].fillna(0)
    
    if 'date' in df_z.index.names:
        df_z = df_z.reset_index(level='date')
        if not isinstance(original_index, pd.MultiIndex):
            df_z = df_z.set_index('date')
    
    return df_z

def get_features_for_ticker(ticker_symbol: str) -> pd.DataFrame:
    try:
        print(f"\n=== Getting features for ticker {ticker_symbol} ===")
        
        get_fundamental_data(ticker_symbol)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=800)
        
        print(f"Fetching historical data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        historical_data = fetch_single_ticker_history(
            ticker_symbol=ticker_symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            interval="1d"
        )
        if historical_data.empty:
            print("Error: Failed to fetch historical price data.")
            return pd.DataFrame()
            
        technical_features = calculate_historical_features_for_ticker(
            df_ohlcv=historical_data,
            ticker_symbol=ticker_symbol
        )
        if technical_features.empty:
            print("Warning: Could not calculate any technical features (history might be too short).")
            technical_features = pd.DataFrame(index=historical_data.index)

        fundamental_features = get_fundamental_features_for_ticker(
            ticker_symbol=ticker_symbol,
            start_date_str=start_date.strftime('%Y-%m-%d')
        )
        if fundamental_features.empty:
            print("Warning: Could not calculate fundamental features.")
            fundamental_features = pd.DataFrame(index=historical_data.index)
        
        fundamental_features.index = pd.to_datetime(fundamental_features.index).normalize()
        technical_features.index = pd.to_datetime(technical_features.index).normalize()

        combined_features = pd.merge(
            technical_features, 
            fundamental_features, 
            left_index=True,
            right_index=True,
            how='left'
        )

        if combined_features.empty:
            print("Error: Failed to combine technical and fundamental data.")
            return pd.DataFrame()

        last_record = combined_features.ffill().iloc[-1:]
        print(f"Successfully generated feature set for prediction.")
        
        return last_record
        
    except Exception as e:
        print(f"An error occurred in get_features_for_ticker: {str(e)}")
        return pd.DataFrame()