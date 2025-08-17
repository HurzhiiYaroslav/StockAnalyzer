import pandas as pd
import numpy as np
import os
import json
import random
from src import config
from src.services import fetch_data
from src.services import process_data

os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

def load_metadata_cache():
    metadata_path = os.path.join(config.DATA_DIR, 'company_metadata.json')
    if not os.path.exists(metadata_path):
        print("Metadata file not found. Please run 'python -m training.fetch_metadata' first.")
        return {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_future_returns_and_ratings_for_combined(all_featured_data, horizon_days, num_quantiles):
    if all_featured_data.empty:
        print("  Combined DataFrame is empty for calculating ratings.")
        return pd.DataFrame()
        
    print(f"  Calculating future returns and ratings...", end=" ", flush=True)
    df = all_featured_data.copy()

    if not (isinstance(df.index, pd.DatetimeIndex) and df.index.name == 'date'):
        print("\nError: Index is not a DateTimeIndex named 'date' for calculating ratings.")
        return pd.DataFrame()
    if 'ticker' not in df.columns:
        print("\nError: 'ticker' column is missing for calculating ratings.")
        return pd.DataFrame()
    if 'close' not in df.columns:
        print("\nError: 'close' column is missing for calculating ratings.")
        return pd.DataFrame()

    df.sort_values(by=['ticker', 'date'], inplace=True)
        
    df['Future_Close'] = df.groupby('ticker')['close'].shift(-horizon_days)
    df['Future_Return'] = (df['Future_Close'] - df['close']) / df['close']
    df.dropna(subset=['Future_Return'], inplace=True)
    
    if df.empty:
        print("\nNo data after calculating Future_Return.")
        return pd.DataFrame()

    df['Rating'] = df.groupby('date')['Future_Return'].transform(
        lambda x: pd.qcut(x, num_quantiles, labels=False, duplicates='drop') + 1 if x.nunique() >= num_quantiles else pd.Series([1]*len(x), index=x.index)
    )
    print("OK")
    return df

def impute_missing_values(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    print("\nStarting intelligent NaN imputation...")
    imputed_df = df.copy()

    technical_cols = [col for col in feature_columns if col.startswith(('SMA', 'RSI', 'ATR', 'BB', 'MACD', 'Stoch', 'ADX', 'Price_vs', 'Return', 'volatility', 'Sharpe', 'Volume', 'OBV'))]
    print(f"  Imputing {len(technical_cols)} technical features by ticker...")
    if technical_cols:
        imputed_df[technical_cols] = imputed_df.groupby('ticker')[technical_cols].transform(
            lambda x: x.ffill().bfill()
        )

    fundamental_cols = [col for col in feature_columns if col not in technical_cols]
    print(f"  Imputing {len(fundamental_cols)} fundamental features by sector-date median (optimized)...")
    
    if fundamental_cols:
        df_for_grouping = imputed_df.reset_index()
        sector_date_medians = df_for_grouping.groupby(['sector', 'date'])[fundamental_cols].median()
        imputed_df[fundamental_cols] = imputed_df.groupby(['sector', 'date'])[fundamental_cols].transform(
            lambda x: x.fillna(x.median()))


    print("  Performing final global forward and backward fill...")
    imputed_df[feature_columns] = imputed_df.groupby('ticker')[feature_columns].ffill().bfill()
    imputed_df[feature_columns] = imputed_df[feature_columns].ffill().bfill()
    
    return imputed_df

def drop_columns_with_high_nan(df: pd.DataFrame, nan_threshold: float = 0.6) -> pd.DataFrame:    
    nan_percentages = df.isnull().sum() / len(df)
    cols_to_drop = nan_percentages[nan_percentages > nan_threshold].index.tolist()
    
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} columns:")
        for col in cols_to_drop:
            print(f"  - {col} ({nan_percentages[col]:.2%} NaN)")
            
        cleaned_df = df.drop(columns=cols_to_drop)
        print(f"DataFrame shape after dropping columns: {cleaned_df.shape}")
        return cleaned_df
    else:
        print("No columns exceeded the NaN threshold. No columns dropped.")
        return df

def main():
    company_metadata = load_metadata_cache()
    if not company_metadata:
        return
    
    individual_stock_data_map = fetch_data.download_historical_data_for_tickers(config.TICKERS, config.START_DATE, config.END_DATE)
    if not individual_stock_data_map:
        print("Failed to download data for any ticker. Exiting.")
        return

    all_featured_dfs_list = []
    for ticker_sym, ohlcv_df in individual_stock_data_map.items():
        print(f"Processing {ticker_sym}:")
        
        technical_features = process_data.calculate_historical_features_for_ticker(ohlcv_df, ticker_sym)
        if technical_features.empty:
            print(f"  Skipping {ticker_sym} due to missing technical features.")
            continue
        
        fund_data = process_data.get_fundamental_features_for_ticker(ticker_sym, config.START_DATE)
        if fund_data.empty:
            print(f"  Skipping {ticker_sym} due to missing fundamental data.")
            continue

        df_fund_to_merge = fund_data.drop(columns=['ticker'], errors='ignore')
        if not isinstance(df_fund_to_merge.index, pd.DatetimeIndex):
            df_fund_to_merge.index = pd.to_datetime(df_fund_to_merge.index)
        
        df_fund_to_merge.index = df_fund_to_merge.index.normalize()
        df_fund_to_merge.index.name = 'date'    
        
        if not isinstance(technical_features.index, pd.DatetimeIndex):
            technical_features.index = pd.to_datetime(technical_features.index)

        technical_features.index = technical_features.index.normalize()
        technical_features.index.name = 'date'

        df_combined_features = pd.merge(
            technical_features, 
            df_fund_to_merge, 
            left_index=True,
            right_index=True,
            how='left'
        )
        
        if not df_combined_features.empty:
            sector = company_metadata.get(ticker_sym, {}).get('sector', 'Unknown')
            df_combined_features['sector'] = sector
            
            if 'close' in df_combined_features.columns and 'ttm_eps' in df_combined_features.columns:
                df_combined_features['P_E_Ratio'] = np.where(
                    (df_combined_features['ttm_eps'].notna()) & (df_combined_features['ttm_eps'] != 0),
                    df_combined_features['close'] / df_combined_features['ttm_eps'], 
                    np.nan
                )
            
            fund_cols_to_fill = list(set(df_fund_to_merge.columns).union(['P_E_Ratio']))
            fund_cols_to_fill = [col for col in fund_cols_to_fill if col in df_combined_features.columns]
            
            if fund_cols_to_fill:
                 df_combined_features[fund_cols_to_fill] = df_combined_features[fund_cols_to_fill].ffill().bfill()
        
        if not df_combined_features.empty:
            all_featured_dfs_list.append(df_combined_features)
    
    if not all_featured_dfs_list:
        print("Failed to calculate/combine features for any ticker. Exiting.")
        return

    combined_features_df = pd.concat(all_featured_dfs_list)
    print(f"All feature DataFrames combined. Total size: {combined_features_df.shape}")

    cleaned_features_df = drop_columns_with_high_nan(combined_features_df)

    print("Calculating sector-based Z-scores for fundamental metrics...")
    
    fundamental_metrics_for_zscore = [
        'P_E_Ratio', 'ttm_eps', 'roa_ttm', 'asset_turnover_ttm',
        'gross_profit_margin_quarterly', 'net_profit_margin_quarterly'
    ]
    
    metrics_to_process = [m for m in fundamental_metrics_for_zscore if m in cleaned_features_df.columns]

    if not isinstance(combined_features_df.index, pd.DatetimeIndex):
         combined_features_df.index = pd.to_datetime(combined_features_df.index)

    if not isinstance(cleaned_features_df.index, pd.DatetimeIndex):
         cleaned_features_df.index = pd.to_datetime(cleaned_features_df.index)

    combined_features_df_with_z = process_data.calculate_sector_z_scores(
        cleaned_features_df, 
        metrics_to_process
    )
    print("Z-scores calculated.")
    
    final_data_with_ratings = calculate_future_returns_and_ratings_for_combined(
        combined_features_df_with_z, 
        config.FUTURE_RETURN_HORIZON_DAYS, 
        config.NUM_CLASSES
    )
    if final_data_with_ratings.empty:
        print("Failed to calculate ratings. Exiting.")
        return

    columns_to_exclude_from_features = [
        'open', 'high', 'low', 'ticker', 'sector',
        'adj_close', 'close',
        'dividends', 'stock_splits', 
        'Future_Close', 'Future_Return'
    ] 
    
    feature_columns = [col for col in final_data_with_ratings.columns 
                       if col not in columns_to_exclude_from_features + ['Rating','date']] 

    training_df_to_save = final_data_with_ratings.reset_index() 
    
    final_columns_to_save = ['date'] + feature_columns + ['Rating']
    final_columns_to_save = [col for col in final_columns_to_save if col in training_df_to_save.columns]

    training_df_final = training_df_to_save[final_columns_to_save].copy()
    
    training_df_final.replace([np.inf, -np.inf], np.nan, inplace=True)

    imputed_df = impute_missing_values(training_df_to_save, feature_columns)

    nan_counts = imputed_df.isnull().sum()
    nan_counts_with_issues = nan_counts[nan_counts > 0]
    if not nan_counts_with_issues.empty:
        print("\nColumns with NaN values AFTER imputation (before final drop):")
        print(nan_counts_with_issues.sort_values(ascending=False))
    else:
        print("\nNo NaN values found after imputation. Great!")

    final_df = imputed_df.dropna()
    columns_to_save = ['date'] + feature_columns + ['Rating']
    final_df_to_save = final_df[columns_to_save]

    if final_df_to_save.empty:
        print("Final training dataset is empty after cleaning.")
        return

    try:
        final_df_to_save.to_csv(config.FINAL_TRAINING_DATA_PATH, index=False)
        print(f"\nFinal dataset ({len(final_df_to_save)} rows) saved to: {config.FINAL_TRAINING_DATA_PATH}")
        print(f"Number of columns in the final file: {len(final_df_to_save.columns)}")
    except Exception as e:
        print(f"Error saving the final dataset: {e}")

if __name__ == "__main__":
    main()