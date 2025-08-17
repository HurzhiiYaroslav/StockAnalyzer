import { useState, useEffect, useCallback } from 'react'
import './App.css'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import Sidebar from './components/Sidebar';
import CompanyInfo from './components/CompanyInfo';
import PriceChart from './components/PriceChart';
import PredictionResult from './components/PredictionResult';
import KeyMetrics from './components/metricsAnalysis';

function App() {
  const [tickers, setTickers] = useState([])
  const [ticker, setTicker] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [companyInfo, setCompanyInfo] = useState(null)
  const [priceHistory, setPriceHistory] = useState(null)
  const [features, setFeatures] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [tickersLoading, setTickersLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')

  const fetchTickers = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/tickers')
      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch tickers')
      }
      
      setTickers(data.tickers)
    } catch (err) {
      setError(err.message)
    }
  }

  const searchTickers = async (query) => {
    if (!query.trim()) {
      await fetchTickers()
      return
    }

    setTickersLoading(true)
    try {
      const response = await fetch(`http://localhost:5000/api/tickers/search?query=${encodeURIComponent(query)}`)
      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to search tickers')
      }
      
      setTickers(data.tickers)
    } catch (err) {
      setError(err.message)
    } finally {
      setTickersLoading(false)
    }
  }

  useEffect(() => {
    fetchTickers()
  }, [])
  useEffect(() => {
    const timer = setTimeout(() => {
      searchTickers(searchQuery)
    }, 300)

    return () => clearTimeout(timer)
  }, [searchQuery])

  const handleTickerSelect = async (selectedTicker) => {
    setTicker(selectedTicker)
    setLoading(true)
    setError(null)
    setPrediction(null)
    setCompanyInfo(null)
    setPriceHistory(null)
    setFeatures(null)
    setActiveTab('overview')

    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ticker: selectedTicker }),
      })

      const data = await response.json()
      console.log(data)
      if (!response.ok) {
        throw new Error(data.error || 'Failed to get prediction')
      }

      setPrediction(data.prediction)
      setCompanyInfo(data.company_info)
      setPriceHistory(data.price_history)
      setFeatures(data.features)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const renderPriceChart = () => {
    if (!priceHistory || priceHistory.length === 0) return null;
    const data = priceHistory.map(([timestamp, price]) => ({
      timestamp: Number(timestamp),
      price: Number(price)
    }));

    const ticks = [];
    if (data.length > 0) {
      const start = new Date(data[0].timestamp);
      const end = new Date(data[data.length - 1].timestamp);
      let year = start.getFullYear();
      let month = start.getMonth();
      while (year < end.getFullYear() || (year === end.getFullYear() && month <= end.getMonth())) {
        ticks.push(new Date(year, month, 1).getTime());
        month++;
        if (month > 11) {
          month = 0;
          year++;
        }
      }
      if (ticks.length > 0) ticks.shift();
    }

    const CustomTooltip = ({ active, payload, label }) => {
      if (active && payload && payload.length) {
        const d = new Date(label);
        return (
          <div style={{ background: 'white', border: '1px solid #eee', padding: 10, borderRadius: 8 }}>
            <div style={{ fontSize: 12, color: '#646cff' }}><b>Date:</b> {d.toLocaleDateString()}</div>
            <div style={{ fontSize: 12, color: '#646cff' }}><b>Price:</b> ${payload[0].value}</div>
          </div>
        );
      }
      return null;
    };

    return (
      <div className="price-chart">
        <h2>Price History (1 Year)</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="timestamp"
              type="number"
              scale="time"
              domain={['dataMin', 'dataMax']}
              ticks={ticks}
              tickFormatter={ts => {
                const d = new Date(ts);
                return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
              }}
              angle={-25}
              textAnchor="end"
              tick={{ fontSize: 10 }}
            />
            <YAxis domain={['auto', 'auto']} tick={{ fontSize: 12 }} />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="price" stroke="#646cff" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  }

  const renderOverviewTab = () => (
    <>
      <CompanyInfo companyInfo={companyInfo} />
      <PriceChart priceHistory={priceHistory} />
      <PredictionResult prediction={prediction} />
    </>
  )

  const featureDisplayNames = {
    'Return_250D': 'Return (1 Year)',
    'Price_vs_SMA200': 'Price vs 200-day SMA',
    'Price_vs_SMA50': 'Price vs 50-day SMA',
    'P_E_Ratio': 'P/E Ratio',
    'ttm_eps': 'EPS (TTM)',
    'Sharpe_Ratio_252D': 'Sharpe Ratio (1 Year)',
    'volatility_250d': 'Volatility (1 Year)',
    'roa_ttm': 'Return on Assets (TTM)',
    'ATR_252D': 'ATR (1 Year)',
    'gross_profit_margin_quarterly': 'Gross Profit Margin (Quarterly)',
    'net_profit_margin_quarterly': 'Net Profit Margin (Quarterly)'
  };

  const getFeatureDisplayName = (feature) => featureDisplayNames[feature] || feature;

  const getFeatureDescription = (feature) => {
    const descriptions = {
      'Return_250D': '250-day return shows the stock\'s performance over the past year.',
      'Price_vs_SMA200': 'Price vs 200-day SMA indicates if the stock is trading above or below its long-term trend.',
      'Price_vs_SMA50': 'Price vs 50-day SMA shows short-term trend strength.',
      'P_E_Ratio': 'Price-to-Earnings ratio measures stock valuation relative to earnings.',
      'ttm_eps': 'Trailing Twelve Months Earnings Per Share shows company profitability.',
      'Sharpe_Ratio_252D': 'Sharpe Ratio measures risk-adjusted returns over the past year.',
      'volatility_250d': '250-day volatility measures price fluctuation risk.',
      'roa_ttm': 'Return on Assets (TTM) shows how efficiently a company uses its assets to generate profit over the trailing twelve months.',
      'ATR_252D': 'Average True Range (252D) measures the average volatility of a stock over the past year (252 trading days).',
      'gross_profit_margin_quarterly': 'Gross Profit Margin (Quarterly) shows the percentage of revenue that exceeds the cost of goods sold for the last quarter.',
      'net_profit_margin_quarterly': 'Net Profit Margin (Quarterly) shows the percentage of revenue left after all expenses for the last quarter.'
    }
    return descriptions[feature] || 'No description available.'
  }

  const renderMetricsTab = () => (
    <KeyMetrics features={features} getFeatureDescription={getFeatureDescription} getFeatureDisplayName={getFeatureDisplayName} />
  )

  return (
    <div className="app-container">
      <Sidebar
        tickers={tickers}
        tickersLoading={tickersLoading}
        error={error}
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        handleTickerSelect={handleTickerSelect}
        selectedTicker={ticker}
      />
      <div className="main-content">
        {loading ? (
          <div className="loading"></div>
        ) : error ? (
          <div className="error-message">{error}</div>
        ) : companyInfo ? (
          <>
            <div className="tabs">
              <button 
                className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
                onClick={() => setActiveTab('overview')}
              >
                Overview
              </button>
              <button 
                className={`tab ${activeTab === 'metrics' ? 'active' : ''}`}
                onClick={() => setActiveTab('metrics')}
              >
                Metrics Analysis
              </button>
            </div>
            <div className="tab-content">
              {activeTab === 'overview' ? renderOverviewTab() : renderMetricsTab()}
            </div>
          </>
        ) : (
          <div className="welcome-message">
            <h2>Welcome to Stock Rating Predictor</h2>
            <p>Select a ticker from the list to view its details and prediction.</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
