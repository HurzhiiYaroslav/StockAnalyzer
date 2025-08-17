import React from 'react';

const Sidebar = ({ tickers, tickersLoading, error, searchQuery, setSearchQuery, handleTickerSelect, selectedTicker }) => (
  <div className="sidebar">
    <h2>Stock List</h2>
    <div className="search-container">
      <input
        type="text"
        value={searchQuery}
        onChange={e => setSearchQuery(e.target.value)}
        placeholder="Search tickers..."
        className="search-input"
      />
    </div>
    <div className="ticker-list">
      {tickersLoading ? (
        <div className="loading" />
      ) : error ? (
        <div className="error-message">{error}</div>
      ) : tickers.length === 0 ? (
        <div className="no-results">No tickers found</div>
      ) : (
        tickers.map(ticker => (
          <div
            key={ticker.symbol}
            className={`ticker-item ${ticker.symbol === selectedTicker ? 'active' : ''}`}
            onClick={() => handleTickerSelect(ticker.symbol)}
          >
            <span className="ticker-name">{ticker.name}</span>
            <span className="ticker-symbol">{ticker.symbol}</span>
          </div>
        ))
      )}
    </div>
  </div>
);

export default Sidebar; 