import React, { useEffect } from 'react';
import { useStore } from '../../store/useStore';
import styles from './Sidebar.module.css';

const Sidebar = () => {
  const { 
    tickers, 
    searchQuery, 
    setSearchQuery,
    tickersLoading,
    tickersError,
    fetchTickers,
    searchTickers,
    selectedTicker,
    fetchPrediction
  } = useStore();

  useEffect(() => {
    fetchTickers();
  }, [fetchTickers]);

  useEffect(() => {
    const timer = setTimeout(() => {
      searchTickers(searchQuery);
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery, searchTickers]);

  return (
    <div className={styles.sidebar}> 
      <div className={styles.sidebarHeader}>
        <h2>Stock List</h2>
        <div className={styles.searchContainer}>
          <input
            type="text"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Search tickers..."
            className={styles.searchInput}
          />
        </div>
      </div>

      <div className={styles.tickerListContainer}> 
        <div className={styles.tickerList}>
          {tickersLoading ? (
            <div className="loading" />
          ) : tickersError ? (
            <div className="error-message">{tickersError}</div>
          ) : tickers.length === 0 ? (
            <div className={styles.noResults}>No tickers found</div>
          ) : (
            tickers.map(ticker => (
              <div
                key={ticker.symbol}
                className={`${styles.tickerItem} ${ticker.symbol === selectedTicker ? styles.active : ''}`}
                onClick={() => fetchPrediction(ticker.symbol)}
              >
                <span className={styles.tickerName}>{ticker.name}</span>
                <span className={styles.tickerSymbol}>{ticker.symbol}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;