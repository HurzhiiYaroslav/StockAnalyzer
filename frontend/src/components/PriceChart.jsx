import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

const PriceChart = ({ priceHistory }) => {
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
          <div style={{ fontSize: 14, color: 'black' }}><b>Date:</b> {d.toLocaleDateString()}</div>
          <div style={{ fontSize: 14, color: 'black' }}><b>Price:</b> ${payload[0].value}</div>
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
              return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
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
};

export default PriceChart; 