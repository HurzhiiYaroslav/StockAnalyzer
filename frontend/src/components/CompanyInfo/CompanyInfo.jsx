import React from 'react';
import styles from './CompanyInfo.module.css';

const CompanyInfo = ({ companyInfo }) => (
  <div className={styles.companyInfo}>
    <div className={styles.companyHeader}>
      <div className={styles.companyMain}>
        <img src={companyInfo.logo_url} alt={`${companyInfo.ticker} logo`} className={styles.companyLogo} />
        <div className={styles.companyTitle}>
          <h2>{companyInfo.name} <span className={styles.tickerSymbol}>({companyInfo.ticker})</span></h2>
          <span className={styles.companySector}>{companyInfo.sector || 'N/A'} - {companyInfo.industry || 'N/A'}</span>
        </div>
      </div>
      <div className={styles.marketData}>
        <div className={styles.marketDataItem}>
          <span className={styles.label}>52W Range:</span>
          <span className={styles.value}>${companyInfo.fifty_two_week_low?.toFixed(2)} - ${companyInfo.fifty_two_week_high?.toFixed(2)}</span>
        </div>
        <div className={styles.marketDataItem}>
          <span className={styles.label}>Day Range:</span>
          <span className={styles.value}>${companyInfo.day_low?.toFixed(2)} - ${companyInfo.day_high?.toFixed(2)}</span>
        </div>
      </div>
    </div>
  </div>
);

export default CompanyInfo;