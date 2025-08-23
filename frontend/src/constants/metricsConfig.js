export const FEATURE_DISPLAY_NAMES = {
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

export const getFeatureDisplayName = (feature) => FEATURE_DISPLAY_NAMES[feature] || feature;

export const FEATURE_DESCRIPTIONS = {
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
};

export const getFeatureDescription = (feature) => FEATURE_DESCRIPTIONS[feature] || 'No description available.';