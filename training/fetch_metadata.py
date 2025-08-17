import json
import os
import time
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src import config
from src.services import fetch_data

METADATA_FILE_PATH = os.path.join(config.DATA_DIR, 'company_metadata.json')

def create_metadata_cache():
    metadata = {}
    if os.path.exists(METADATA_FILE_PATH):
        with open(METADATA_FILE_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"Loaded {len(metadata)} records from existing metadata cache.")

    tickers_to_fetch = [t for t in config.TICKERS if t not in metadata]
    
    if not tickers_to_fetch:
        print("Metadata cache is already up to date.")
        return

    print(f"Fetching metadata for {len(tickers_to_fetch)} new tickers...")
    
    for i, ticker in enumerate(tickers_to_fetch):
        print(f"({i+1}/{len(tickers_to_fetch)}) Fetching info for {ticker}...")
        info = fetch_data.get_company_info(ticker)
        if info:
            metadata[ticker] = {
                "name": info.get("name"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown")
            }
        else:
            metadata[ticker] = {
                "name": ticker,
                "sector": "Unknown",
                "industry": "Unknown"
            }
        time.sleep(0.5)

    with open(METADATA_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
        
    print(f"Successfully saved metadata for {len(metadata)} tickers to {METADATA_FILE_PATH}")

if __name__ == "__main__":
    create_metadata_cache()