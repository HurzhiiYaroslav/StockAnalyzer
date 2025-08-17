import React from 'react';

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

const PredictionResult = ({ prediction }) => (
  prediction !== null && (
    <div className="prediction-result">
      <div className="rating-display">
        <span className="rating-value" data-rating={prediction}>
          Rating: {prediction}
        </span>
        <span className="rating-explanation">
          {getRatingExplanation(prediction)}
        </span>
      </div>
    </div>
  )
);

export default PredictionResult; 