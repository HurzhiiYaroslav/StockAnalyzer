import React from 'react';
import styles from './MetricsAnalysis.module.css';

const METRIC_EVALUATION_RULES = {
  P_E_Ratio: {
    evaluate: (v) => (v < 15 ? 'Undervalued' : v < 25 ? 'Fair Value' : 'Overvalued'),
    getColor: (v) => (v < 15 ? styles.positive : v < 25 ? styles.neutral : styles.negative),
  },
  Price_vs_SMA200D: {
    evaluate: (v) => (v > 1.1 ? 'Strong Uptrend' : v > 1 ? 'Above Trend' : 'Below Trend'),
    getColor: (v) => (v > 1.1 ? styles.positive : v > 1 ? styles.neutral : styles.negative),
  },
  Price_vs_SMA50D: {
    evaluate: (v) => (v > 1.1 ? 'Strong Momentum' : v > 1 ? 'Positive' : 'Negative'),
    getColor: (v) => (v > 1.1 ? styles.positive : v > 1 ? styles.neutral : styles.negative),
  },
  Return_252D: {
    evaluate: (v) => (v > 0.2 ? 'Excellent' : v > 0 ? 'Positive' : 'Negative'),
    getColor: (v) => (v > 0.2 ? styles.positive : v > 0 ? styles.neutral : styles.negative),
  },
  Sharpe_Ratio_252D: {
    evaluate: (v) => (v > 1 ? 'Good Risk-Adjusted Return' : v > 0 ? 'Acceptable' : 'Poor'),
    getColor: (v) => (v > 1 ? styles.positive : v > 0 ? styles.neutral : styles.negative),
  },
  ttm_eps: {
    evaluate: (v) => (v > 0 ? 'Profitable' : 'Unprofitable'),
    getColor: (v) => (v > 0 ? styles.positive : styles.negative),
  },
  volatility_252D: {
    evaluate: (v) => (v < 0.2 ? 'Low Risk' : v < 0.4 ? 'Moderate Risk' : 'High Risk'),
    getColor: (v) => (v < 0.2 ? styles.positive : v < 0.4 ? styles.neutral : styles.negative),
  },
  roa_ttm: {
    evaluate: (v) => (v > 0.1 ? 'Excellent' : v > 0.05 ? 'Good' : 'Low'),
    getColor: (v) => (v > 0.1 ? styles.positive : v > 0.05 ? styles.neutral : styles.negative),
  },
  ATR_252D: {
    evaluate: (v) => (v < 0.15 ? 'Low Volatility' : v < 0.3 ? 'Moderate Volatility' : 'High Volatility'),
    getColor: (v) => (v < 0.15 ? styles.positive : v < 0.3 ? styles.neutral : styles.negative),
  },
  gross_profit_margin_quarterly: {
    evaluate: (v) => (v > 0.4 ? 'High Margin' : v > 0.2 ? 'Moderate Margin' : 'Low Margin'),
    getColor: (v) => (v > 0.4 ? styles.positive : v > 0.2 ? styles.neutral : styles.negative),
  },
  net_profit_margin_quarterly: {
    evaluate: (v) => (v > 0.2 ? 'High Profitability' : v > 0.1 ? 'Moderate Profitability' : 'Low Profitability'),
    getColor: (v) => (v > 0.2 ? styles.positive : v > 0.1 ? styles.neutral : styles.negative),
  },
  default: {
    evaluate: (v) => (v > 0 ? 'Good' : 'Bad'),
    getColor: (v) => (v > 0 ? styles.positive : styles.negative),
  },
};

const MetricItem = ({ metricKey, value, getFeatureDisplayName, getFeatureDescription }) => {
  const rules = METRIC_EVALUATION_RULES[metricKey] || METRIC_EVALUATION_RULES.default;
  const isNumeric = typeof value === 'number' && !isNaN(value);

  const evaluationText = isNumeric ? rules.evaluate(value) : '';
  const colorClass = isNumeric ? rules.getColor(value) : '';

  return (
    <div className={styles.featureItem}>
      <div className={styles.featureContent}>
        <span className={styles.featureName}>{getFeatureDisplayName(metricKey)}</span>
        <span className={`${styles.featureValue} ${colorClass}`}>
          {isNumeric ? value.toFixed(4) : String(value)}
        </span>
        {evaluationText && (
          <span className={`${styles.featureEvaluation} ${colorClass}`}>
            ({evaluationText})
          </span>
        )}
        <div className={styles.featureInfo}>
          <svg className={styles.infoIcon} viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="16" x2="12" y2="12" />
            <line x1="12" y1="8" x2="12.01" y2="8" />
          </svg>
          <div className={styles.tooltip}>
            {getFeatureDescription(metricKey)}
          </div>
        </div>
      </div>
    </div>
  );
};

const KeyMetrics = ({ features, getFeatureDescription, getFeatureDisplayName }) => {
  const hasData = Array.isArray(features) ? features.length > 0 : features && Object.keys(features).length > 0;
  const metricsObject = Array.isArray(features) ? features[0] : features;
  
  return (
    <div className={styles.metricsAnalysis}>
      <h2>Key Metrics</h2>
      {hasData ? (
        <div className={styles.featuresList}>
          {Object.entries(metricsObject)
            .filter(([_, value]) => value !== null && value !== undefined)
            .map(([key, value]) => (
              <MetricItem
                key={key}
                metricKey={key}
                value={value}
                getFeatureDisplayName={getFeatureDisplayName}
                getFeatureDescription={getFeatureDescription}
              />
            ))}
        </div>
      ) : (
        <p>No key metrics data available.</p>
      )}
    </div>
  );
};

export default KeyMetrics;