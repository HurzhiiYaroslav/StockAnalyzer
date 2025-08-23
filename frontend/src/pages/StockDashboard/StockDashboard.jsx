import React, { useState, useEffect } from 'react';
import styles from './StockDashboard.module.css';
import { useStore } from '../../store/useStore';
import CompanyInfo from '../../components/CompanyInfo/CompanyInfo';
import PriceChart from '../../components/PriceChart/PriceChart';
import PredictionResult from '../../components/PredictionResult/PredictionResult';
import KeyMetrics from '../../components/MetricsAnalysis/MetricsAnalysis';
import { getFeatureDescription, getFeatureDisplayName } from '../../constants/metricsConfig';

const StockDashboard = () => { 
  const { 
    predictionData, 
    predictionLoading, 
    predictionError,
    selectedTicker 
  } = useStore();
  
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    setActiveTab('overview');
  }, [selectedTicker]);

  if (predictionLoading) {
    return <div className="loading"></div>;
  }
  
  if (predictionError) {
    return <div className="error-message">{predictionError}</div>;
  }

  if (!predictionData || !selectedTicker) {
    return (
      <div className="welcome-message">
        <h2>Welcome to Stock Rating Predictor</h2>
        <p>Select a ticker from the list to view its details and prediction.</p>
      </div>
    );
  }

  const { prediction, company_info, price_history } = predictionData;

  const renderOverviewTab = () => (
    <>
      <CompanyInfo companyInfo={company_info} />
      <PriceChart priceHistory={price_history} />
      <PredictionResult prediction={prediction.predicted_rating} />
    </>
  );

  const renderMetricsTab = () => (
    <KeyMetrics 
      features={prediction.features_for_display ? [prediction.features_for_display] : []}
      getFeatureDescription={getFeatureDescription} 
      getFeatureDisplayName={getFeatureDisplayName} 
    />
  );
  
  return (
    <>
      <div className={styles.tabs}>
        <button 
          className={`${styles.tab} ${activeTab === 'overview' ? styles.active : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button 
          className={`${styles.tab} ${activeTab === 'metrics' ? styles.active : ''}`}
          onClick={() => setActiveTab('metrics')}
        >
          Metrics Analysis
        </button>
      </div>
      <div className={styles.tabContent}>
        {activeTab === 'overview' ? renderOverviewTab() : renderMetricsTab()}
      </div>
    </>
  );
};

export default StockDashboard;