import pandas as pd
import numpy as np
import os
from src import config
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.services.feature_engineering import generate_features_for_ticker

os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

def calculate_future_returns_and_ratings_for_combined(all_featured_data, horizon_days, num_quantiles):
    if all_featured_data.empty:
        print("  Combined DataFrame is empty for calculating ratings.")
        return pd.DataFrame()
    print(f"  Calculating future returns and ratings...", end=" ", flush=True)
    df = all_featured_data.copy()
    if df.index.name == 'date':
        df = df.reset_index()
    if 'date' not in df.columns or 'ticker' not in df.columns or 'close' not in df.columns:
        print("\nError: Required columns are missing for rating calculation.")
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
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

def _impute_ticker_chunk(df_chunk: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Helper function to perform imputation on a single ticker's DataFrame chunk."""
    df_chunk[feature_columns] = df_chunk[feature_columns].ffill()
    return df_chunk

def impute_missing_values(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Performs intelligent NaN imputation on the dataset, parallelizing per-ticker operations."""
    print("\nStarting intelligent NaN imputation...")
    imputed_df = df.copy()
    
    technical_cols = [col for col in feature_columns if col.startswith(config.TECHNICAL_INDICATOR_PREFIXES)]
    fundamental_cols = [col for col in feature_columns if col not in technical_cols]
    
    print(f"  Imputing {len(fundamental_cols)} fundamental features by sector-date median...")
    if fundamental_cols:
        df_for_grouping = imputed_df if 'date' in imputed_df.columns else imputed_df.reset_index()
        imputed_df[fundamental_cols] = df_for_grouping.groupby(['sector', 'date'])[fundamental_cols].transform(
            lambda x: x.fillna(x.median()) if not x.isnull().all() else x)

    print(f"  Imputing remaining features in parallel across {os.cpu_count()} cores...")
    ticker_groups = [group for _, group in imputed_df.groupby('ticker')]
    imputed_chunks = []
    
    with ProcessPoolExecutor(max_workers=None) as executor:
        future_to_chunk = {
            executor.submit(_impute_ticker_chunk, chunk, feature_columns): chunk for chunk in ticker_groups
        }
        for future in as_completed(future_to_chunk):
            try:
                imputed_chunks.append(future.result())
            except Exception as e:
                print(f"A chunk failed to process: {e}")

    if not imputed_chunks:
        print("Error: No chunks were processed successfully.")
        return pd.DataFrame()
    imputed_df = pd.concat(imputed_chunks).sort_index()

    print("  Performing final zero fill...")
    imputed_df[feature_columns] = imputed_df[feature_columns].fillna(0)
    return imputed_df

def drop_columns_with_high_nan(df: pd.DataFrame, nan_threshold: float = 0.6) -> pd.DataFrame:    
    nan_percentages = df.isnull().sum() / len(df)
    cols_to_drop = nan_percentages[nan_percentages > nan_threshold].index.tolist()
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} columns with NaN > {nan_threshold:.0%}: {cols_to_drop}")
        cleaned_df = df.drop(columns=cols_to_drop)
        print(f"DataFrame shape after dropping columns: {cleaned_df.shape}")
        return cleaned_df
    else:
        print("No columns exceeded the NaN threshold.")
        return df

def main():    
    tickers_to_process = config.DEF_TICKERS
    all_featured_dfs_list = []
    all_fundamental_metrics = set() 

    with ProcessPoolExecutor(max_workers=None) as executor:
        future_to_ticker = {
            executor.submit(generate_features_for_ticker, ticker, config.START_DATE, config.END_DATE): ticker 
            for ticker in tickers_to_process
        }
        print(f"Submitted {len(tickers_to_process)} tickers for parallel processing.")
        for i, future in enumerate(as_completed(future_to_ticker)):
            ticker = future_to_ticker[future]
            print(f"({i+1}/{len(tickers_to_process)}) Processing completed for ticker: {ticker}")
            try:
                df_features, fundamental_metrics = future.result()
                if not df_features.empty:
                    all_featured_dfs_list.append(df_features)
                    all_fundamental_metrics.update(fundamental_metrics)
            except Exception as exc:
                print(f"  Ticker {ticker} generated an exception: {exc}")
    
    if not all_featured_dfs_list:
        print("Failed to generate features for any ticker. Exiting.")
        return
    
    combined_features_df = pd.concat(all_featured_dfs_list)
    print(f"All feature DataFrames combined. Total size: {combined_features_df.shape}")

    cleaned_features_df = drop_columns_with_high_nan(combined_features_df)

    final_data_with_ratings = calculate_future_returns_and_ratings_for_combined(
        cleaned_features_df, config.FUTURE_RETURN_HORIZON_DAYS, config.NUM_CLASSES
    )

    if final_data_with_ratings.empty:
        print("Failed to calculate ratings. Exiting.")
        return

    columns_to_exclude = [
        'open', 'high', 'low', 'adj_close', 'close', 'dividends', 
        'stock_splits', 'Future_Close', 'Future_Return'
    ]
    
    feature_columns = [
        col for col in final_data_with_ratings.columns 
        if col not in columns_to_exclude and col not in ['date', 'ticker', 'sector', 'Rating']
    ]
    
    imputed_df = impute_missing_values(final_data_with_ratings, feature_columns)
    imputed_df.dropna(subset=['Rating'], inplace=True)
    
    columns_to_save = ['date', 'ticker', 'sector', 'Rating'] + feature_columns
    final_df_to_save = imputed_df[columns_to_save]

    final_df_to_save.to_csv(config.FINAL_TRAINING_DATA_PATH, index=False)
    print(f"\nFinal dataset ({len(final_df_to_save)} rows) saved to: {config.FINAL_TRAINING_DATA_PATH}")

if __name__ == "__main__":
    main()