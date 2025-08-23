import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import requests
from src import config

def fetch_single_ticker_history(ticker_symbol: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
    """Fetches historical OHLCV data for a single ticker, using a local cache to minimize API calls."""
    os.makedirs(config.PRICE_CACHE_DIR, exist_ok=True)
    cache_file_path = os.path.join(config.PRICE_CACHE_DIR, f"{ticker_symbol}.feather")

    data_to_return = pd.DataFrame()

    def process_df(df):
        if not df.empty:
            df.columns = df.columns.str.lower()
            if 'adj close' in df.columns:
                df['close'] = df['adj close']
            
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert('UTC')
            
            df.index = df.index.tz_localize(None).normalize()
            df.index.name = 'date'
        return df

    try:
        if os.path.exists(cache_file_path):
            cached_data = pd.read_feather(cache_file_path).set_index('date')
            if not cached_data.empty:
                last_cached_date = cached_data.index.max()
                print(f"  Cache found for {ticker_symbol}. Last date: {last_cached_date.strftime('%Y-%m-%d')}", end=". ")

                if last_cached_date < (pd.to_datetime(end_date) - timedelta(days=2)):
                    print("Updating cache...")
                    new_start_date = (last_cached_date + timedelta(days=1)).strftime('%Y-%m-%d')
                    new_data_raw = yf.Ticker(ticker_symbol).history(start=new_start_date, end=end_date, interval=interval, auto_adjust=False)
                    new_data = process_df(new_data_raw)
                    if not new_data.empty:
                        updated_data = pd.concat([cached_data, new_data])
                        updated_data = updated_data[~updated_data.index.duplicated(keep='last')]
                        updated_data.reset_index().to_feather(cache_file_path)
                        data_to_return = updated_data
                    else:
                        print("No new data to add.")
                        data_to_return = cached_data
                else:
                    print("Cache is up to date.")
                    data_to_return = cached_data
            else:
                os.remove(cache_file_path)
                return fetch_single_ticker_history(ticker_symbol, start_date, end_date, interval)
        else:
            print(f"  No cache for {ticker_symbol}. Downloading full history...", end=" ")
            full_data_raw = yf.Ticker(ticker_symbol).history(start=start_date, end=end_date, interval=interval, auto_adjust=False)
            full_data = process_df(full_data_raw)
            if not full_data.empty:
                full_data.reset_index().to_feather(cache_file_path)
                data_to_return = full_data
            print("Done.")

        if not data_to_return.empty:
            return data_to_return.loc[start_date:end_date].copy()
        
    except Exception as e:
        print(f"Error fetching/caching historical data for {ticker_symbol}: {str(e)}")
    return pd.DataFrame()

def get_fundamental_data(ticker_symbol: str, period: str = "Q") -> bool:
    """Downloads and saves raw fundamental data for a single ticker from the DataJockey API."""
    if not config.DATAJOCKEY_API_KEY:
        print(f"DataJockey API key not specified.")
        return False

    params = { "apikey": config.DATAJOCKEY_API_KEY, "filetype": "json", "period": period, "ticker": ticker_symbol.upper() }
    time.sleep(13)

    try:
        response = requests.get(config.DATAJOCKEY_BASE_URL, params=params, timeout=30)
        if response.status_code == 429:
            print(f"    ERROR: Rate limit hit (429) for {ticker_symbol}. Stopping.")
            return False
        response.raise_for_status()
        
        os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
        file_path = os.path.join(config.RAW_DATA_DIR, f"{ticker_symbol}.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
            print(f"    Successfully saved data for {ticker_symbol}.")
            return True
    except requests.exceptions.RequestException as e:
        print(f"    An error occurred for {ticker_symbol}: {e}")
        return False

def get_company_info(ticker_symbol: str) -> dict:
    """Retrieves general company information from yfinance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        company_data = {
            "name": info.get("longName"), "ticker": ticker_symbol.upper(),
            "logo_url": f"https://img.logo.dev/ticker/{ticker_symbol.upper()}?token={config.LOGO_API_KEY}",
            "sector": info.get("sector"), "industry": info.get("industry"), "market_cap": info.get("marketCap"),
            "dividend_yield": info.get("dividendYield"), "trailing_pe": info.get("trailingPE"),
            "trailing_eps": info.get("trailingEps"), "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"), "day_low": info.get("dayLow"),
            "day_high": info.get("dayHigh"), "previous_close": info.get("previousClose"),
            "open": info.get("open"), "volume": info.get("volume"), "average_volume": info.get("averageVolume"),
            "current_price": info.get("currentPrice", info.get("regularMarketPrice"))
        }
        return company_data
    except Exception:
        return None

def get_price_history_for_chart(ticker_symbol: str, days_back: int = 365) -> list:
    """Fetches and formats historical price data specifically for charting purposes."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back + 10)
        hist = yf.Ticker(ticker_symbol).history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval="1d")
        if not hist.empty:
            hist_processed = hist[['Close']].dropna()
            chart_data = [[int(idx.timestamp() * 1000), round(row['Close'], 2)] for idx, row in hist_processed.iterrows()]
            return chart_data[-days_back:]
        return []
    except Exception:
        return []