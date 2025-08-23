import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import styles from './PriceChart.module.css';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const d = new Date(label);
    return (
      <div className={styles.tooltip}>
        <div className={styles.tooltipLabel}><b>Date:</b> {d.toLocaleDateString()}</div>
        <div className={styles.tooltipValue}><b>Price:</b> ${payload[0].value.toFixed(2)}</div>
      </div>
    );
  }
  return null;
};

const PriceChart = ({ priceHistory }) => {
  if (!priceHistory || priceHistory.length === 0) {
    return (
      <div className={styles.priceChart}>
        <h2>Price History (1 Year)</h2>
        <p>No price data available.</p>
      </div>
    );
  }

  const data = priceHistory.map(([timestamp, price]) => ({
    timestamp: Number(timestamp),
    price: Number(price)
  }));

  const ticks = [];
  if (data.length > 0) {
    const start = new Date(data[0].timestamp);
    const end = new Date(data[data.length - 1].timestamp);
    let current = new Date(start.getFullYear(), start.getMonth(), 1);

    while (current <= end) {
      ticks.push(current.getTime());
      current.setMonth(current.getMonth() + 1);
    }
    if (ticks.length > 0) ticks.shift();
  }

  return (
    <div className={styles.priceChart}>
      <h2>Price History (1 Year)</h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="timestamp"
            type="number"
            scale="time"
            domain={['dataMin', 'dataMax']}
            ticks={ticks}
            tickFormatter={ts => {
              const d = new Date(ts);
              return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short' });
            }}
            angle={-25}
            textAnchor="end"
            tick={{ fontSize: 10 }}
            height={40}
          />
          <YAxis domain={['auto', 'auto']} tick={{ fontSize: 12 }} />
          <Tooltip content={<CustomTooltip />} />
          <Line type="monotone" dataKey="price" stroke="#646cff" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PriceChart;