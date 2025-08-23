# Stock Performance Predictor

This project uses a deep learning model to predict the future performance of stocks, ranking them on a scale based on a combination of technical and fundamental analysis. The system is capable of automatically fetching, processing, and analyzing financial data to train a neural network and serve predictions via a REST API.

## Features

- **Automated Data Pipeline**: Fetches historical prices and company fundamentals.
- **Advanced Feature Engineering**: Calculates over 150 technical and fundamental indicators, including sector-relative Z-scores to provide market context.
- **AI-Powered Ranking**: A custom Neural Network, optimized with Keras Tuner, learns to rank stocks based on their one-year forward returns.
- **REST API for Predictions**: A Flask-based backend serves model predictions, complete with company info and key financial metrics.

## API Key Configuration

This project requires API keys to function correctly.

1.  Create a file named `.env` in the root directory of the project.
2.  Add your API keys to this file in the following format:

    ```
    DATAJOCKEY_API_KEY="your_datajockey_api_key_here"
    LOGO_API_KEY="your_logodev_api_key_here"
    ```

## Usage

The project workflow is divided into two main parts: training the model and running the prediction API.

### 1. Training the Model

Execute the following commands from the project's root directory to prepare the data and train the model.

**Step 1: Fetch Company Metadata**
This script creates a local cache of company metadata (sector, industry). Run it once or to update the cache.
```bash
python -m training.fetch_metadata
```
**Step 2: Prepare the Training Dataset**
This script builds the complete dataset with all features. This is a long-running process.
```bash
python -m training.prepare_data
```
**3: Train the Neural Network**
This will start the hyperparameter search and train the final model. Artifacts will be saved to backend/ml_inference/model/.
```bash
python -m training.train_neural
```

### 2. Running the Application (Frontend & Backend)

After a model has been successfully trained, you can easily start both the backend and frontend servers using the provided script.

**Quick Start:**
Simply run the start_app.bat file from the project's root directory.
```bash
.\start_app.bat
```
This script will:
- Activate the Python virtual environment.
- Start the Flask backend server in a new window (available at http://localhost:5000).
- Start the frontend development server in another new window (available at http://localhost:5173).


**Manual Start:**

Backend:
```bash
python -m backend.app
```
Frontend:
```bash
cd frontend && npm run dev
```

### Technologies Used
- **Backend & Machine Learning:** Python, TensorFlow / Keras, Pandas, NumPy, Scikit-learn, Flask
- **Frontend:** React, Vite, Recharts
- **Data Sources:** yfinance, DataJockey API
