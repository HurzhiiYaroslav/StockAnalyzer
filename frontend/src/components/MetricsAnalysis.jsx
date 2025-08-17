import React from 'react';
import './MetricsAnalysis.css';

const KeyMetrics = ({ features, getFeatureDescription, getFeatureDisplayName }) => {
  const getEvaluationColor = (key, value) => {
    if (typeof value !== 'number') return '';
    
    switch(key) {
      case 'P_E_Ratio':
        return value < 15 ? 'positive' : value < 25 ? 'neutral' : 'negative';
      
      case 'Price_vs_SMA200':
        return value > 1.1 ? 'positive' : value > 1 ? 'neutral' : 'negative';
      
      case 'Price_vs_SMA50':
        return value > 1.1 ? 'positive' : value > 1 ? 'neutral' : 'negative';
      
      case 'Return_250D':
        return value > 0.2 ? 'positive' : value > 0 ? 'neutral' : 'negative';
      
      case 'Sharpe_Ratio_252D':
        return value > 1 ? 'positive' : value > 0 ? 'neutral' : 'negative';
      
      case 'ttm_eps':
        return value > 0 ? 'positive' : 'negative';
      
      case 'volatility_250d':
        return value < 0.2 ? 'positive' : value < 0.4 ? 'neutral' : 'negative';
      
      case 'roa_ttm':
        return value > 0.1 ? 'positive' : value > 0.05 ? 'neutral' : 'negative';
      
      case 'ATR_252D':
        return value < 0.15 ? 'positive' : value < 0.3 ? 'neutral' : 'negative';
      
      case 'gross_profit_margin_quarterly':
        return value > 0.4 ? 'positive' : value > 0.2 ? 'neutral' : 'negative';
      
      case 'net_profit_margin_quarterly':
        return value > 0.2 ? 'positive' : value > 0.1 ? 'neutral' : 'negative';
      
      default:
        return value > 0 ? 'positive' : 'negative';
    }
  };

  const evaluateValue = (key, value) => {
    if (typeof value !== 'number') return '';
    
    switch(key) {
      case 'P_E_Ratio':
        return value < 15 ? 'Undervalued' : value < 25 ? 'Fair Value' : 'Overvalued';
      
      case 'Price_vs_SMA200':
        return value > 1.1 ? 'Strong Uptrend' : value > 1 ? 'Above Trend' : 'Below Trend';
      
      case 'Price_vs_SMA50':
        return value > 1.1 ? 'Strong Momentum' : value > 1 ? 'Positive' : 'Negative';
      
      case 'Return_250D':
        return value > 0.2 ? 'Excellent' : value > 0 ? 'Positive' : 'Negative';
      
      case 'Sharpe_Ratio_252D':
        return value > 1 ? 'Good Risk-Adjusted Return' : value > 0 ? 'Acceptable' : 'Poor';
      
      case 'ttm_eps':
        return value > 0 ? 'Profitable' : 'Unprofitable';
      
      case 'volatility_250d':
        return value < 0.2 ? 'Low Risk' : value < 0.4 ? 'Moderate Risk' : 'High Risk';
      
      case 'roa_ttm':
        return value > 0.1 ? 'Excellent' : value > 0.05 ? 'Good' : 'Low';
      
      case 'ATR_252D':
        return value < 0.15 ? 'Low Volatility' : value < 0.3 ? 'Moderate Volatility' : 'High Volatility';
      
      case 'gross_profit_margin_quarterly':
        return value > 0.4 ? 'High Margin' : value > 0.2 ? 'Moderate Margin' : 'Low Margin';
      
      case 'net_profit_margin_quarterly':
        return value > 0.2 ? 'High Profitability' : value > 0.1 ? 'Moderate Profitability' : 'Low Profitability';
      
      default:
        return value > 0 ? 'Good' : 'Bad';
    }
  };

  return (
    <div className="metrics-analysis">
      <h2>Key Metrics</h2>
      {features && features.length > 0 && (
        <div className="features-list">
          {Object.entries(features[0])
            .filter(([_, value]) => value !== null && value !== undefined && !(typeof value === 'number' && isNaN(value)))
            .map(([key, value]) => (
              <div key={key} className="feature-item">
                <div className="feature-content">
                  <span className="feature-name">{getFeatureDisplayName(key)}</span>
                  <span className={`feature-value ${getEvaluationColor(key, value)}`}>
                    {typeof value === 'number' ? value.toFixed(4) : value}
                  </span>
                  <span className={`feature-evaluation ${getEvaluationColor(key, value)}`}>
                    ({evaluateValue(key, value)})
                  </span>
                  <div className="feature-info">
                    <svg 
                      className="info-icon" 
                      viewBox="0 0 24 24" 
                      width="16" 
                      height="16" 
                      fill="none" 
                      stroke="currentColor" 
                      strokeWidth="2" 
                      strokeLinecap="round" 
                      strokeLinejoin="round"
                    >
                      <circle cx="12" cy="12" r="10" />
                      <line x1="12" y1="16" x2="12" y2="12" />
                      <line x1="12" y1="8" x2="12.01" y2="8" />
                    </svg>
                    <div className="tooltip">
                      {getFeatureDescription(key)}
                    </div>
                  </div>
                </div>
              </div>
            ))}
        </div>
      )}
    </div>
  );
};

export default KeyMetrics;
