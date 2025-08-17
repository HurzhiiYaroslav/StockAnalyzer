import React from 'react';

const CompanyInfo = ({ companyInfo }) => (
  <div className="company-info">
    <div className="company-header">
      <div className="company-main">
        <img src={companyInfo.logo_url} alt={`${companyInfo.ticker} logo`} className="company-logo" />
        <div className="company-title">
          <h2>{companyInfo.name} <span className="ticker-symbol">({companyInfo.ticker})</span></h2>
          <span className="company-sector">{companyInfo.sector || 'N/A'} - {companyInfo.industry || 'N/A'}</span>
        </div>
      </div>
      <div className="market-data">
        <div className="market-data-item">
          <span className="label">52W Range:</span>
          <span className="value">${companyInfo.fifty_two_week_low?.toFixed(2)} - ${companyInfo.fifty_two_week_high?.toFixed(2)}</span>
        </div>
        <div className="market-data-item">
          <span className="label">Day Range:</span>
          <span className="value">${companyInfo.day_low?.toFixed(2)} - ${companyInfo.day_high?.toFixed(2)}</span>
        </div>
      </div>
    </div>
  </div>
);

export default CompanyInfo; 