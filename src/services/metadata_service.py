import os
import json
import time
from src import config
from src.services.fetch_data import get_company_info

METADATA_FILE_PATH = os.path.join(config.DATA_DIR, 'company_metadata.json')

def load_cache() -> dict:
    """Loads the entire company metadata cache from a JSON file."""
    if not os.path.exists(METADATA_FILE_PATH):
        return {}
    try:
        with open(METADATA_FILE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading metadata cache file: {e}")
        return {}

def _save_cache(metadata: dict):
    """Saves the metadata cache to a JSON file."""
    try:
        os.makedirs(os.path.dirname(METADATA_FILE_PATH), exist_ok=True)
        with open(METADATA_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
    except IOError as e:
        print(f"Error saving metadata cache: {e}")

def get_for_ticker(ticker_symbol: str, update_cache_file: bool = True) -> dict:
    """Retrieves metadata for a single ticker using a cache-aside strategy."""
    ticker_symbol = ticker_symbol.upper()
    metadata_cache = load_cache()

    if ticker_symbol in metadata_cache:
        return metadata_cache[ticker_symbol]

    print(f"  Metadata for {ticker_symbol} not in cache. Fetching from API...")
    info = get_company_info(ticker_symbol)
    
    ticker_metadata = {}
    if info and info.get("sector"):
        ticker_metadata = {"name": info.get("name"), "sector": info.get("sector", "Unknown"), "industry": info.get("industry", "Unknown")}
    else:
        ticker_metadata = {"name": info.get("name") if info else ticker_symbol, "sector": "Unknown", "industry": "Unknown"}
    
    metadata_cache[ticker_symbol] = ticker_metadata
    
    if update_cache_file:
        _save_cache(metadata_cache)
        
    return ticker_metadata

def populate_full_cache():
    """Populates or updates the metadata cache for all tickers specified in config."""
    all_tickers = config.TICKERS
    if not all_tickers:
        print("No tickers specified in config.TICKERS. Exiting cache population.")
        return

    print(f"--- Starting metadata cache population for {len(all_tickers)} tickers ---")
    
    existing_metadata = load_cache()
    tickers_to_fetch = [t for t in all_tickers if t.upper() not in existing_metadata]
    
    if not tickers_to_fetch:
        print("Metadata cache is already up to date for all configured tickers.")
        return
        
    print(f"Found {len(tickers_to_fetch)} new tickers to fetch metadata for.")
    total_to_fetch = len(tickers_to_fetch)

    for i, ticker in enumerate(tickers_to_fetch):
        print(f"({i+1}/{total_to_fetch}) Fetching and caching info for {ticker}...")
        get_for_ticker(ticker, update_cache_file=True)
        if (i + 1) < total_to_fetch:
            time.sleep(0.5)

    print("\n--- Metadata cache population complete! ---")