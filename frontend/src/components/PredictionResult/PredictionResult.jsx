import React from 'react';
import styles from './PredictionResult.module.css';

const getRatingExplanation = (rating) => {
  const explanations = {
    1: "Strong Sell - High risk of significant price decline",
    2: "Sell - Expected to underperform the market",
    3: "Hold - Expected to perform in line with the market",
    4: "Buy - Expected to outperform the market",
    5: "Strong Buy - High potential for significant price increase"
  };
  return explanations[rating] || "Rating not available";
};

const PredictionResult = ({ prediction }) => {
  if (prediction === null || typeof prediction === 'undefined') {
    return (
      <div className={styles.predictionResult}>
        <h2>Prediction</h2>
        <p>Loading prediction...</p>
      </div>
    );
  }
  const rating = Number(prediction);

  return (
    <div className={styles.predictionResult}>
      <h2>Prediction</h2>
      <div className={styles.ratingDisplay}>
        <span className={styles.ratingValue} data-rating={rating}>
          Rating: {rating}
        </span>
        <span className={styles.ratingExplanation}>
          {getRatingExplanation(rating)}
        </span>
      </div>
    </div>
  );
};

export default PredictionResult;