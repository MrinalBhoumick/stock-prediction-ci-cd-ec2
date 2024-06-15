import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import streamlit as st
import threading
import uvicorn
import requests

# Constants
START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Predefined list of stock tickers with their names
STOCK_TICKERS = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com, Inc.',
    'TSLA': 'Tesla, Inc.',
    'META': 'Meta Platforms, Inc.',
    'BRK-B': 'Berkshire Hathaway Inc.',
    'JPM': 'JPMorgan Chase & Co.',
    'V': 'Visa Inc.',
    'JNJ': 'Johnson & Johnson',
    'WMT': 'Walmart Inc.',
    'NVDA': 'NVIDIA Corporation',
    'PG': 'Procter & Gamble Co.',
    'DIS': 'The Walt Disney Company',
    'MA': 'Mastercard Incorporated',
    'HD': 'The Home Depot, Inc.',
    'UNH': 'UnitedHealth Group Incorporated',
    'VZ': 'Verizon Communications Inc.',
    'ADBE': 'Adobe Inc.',
    'NFLX': 'Netflix, Inc.',
    'PYPL': 'PayPal Holdings, Inc.',
    'INTC': 'Intel Corporation',
    'PFE': 'Pfizer Inc.',
    'KO': 'The Coca-Cola Company',
    'MRK': 'Merck & Co., Inc.'
}

# FastAPI setup
app = FastAPI()

class TickerData(BaseModel):
    ticker: str

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

def preprocess_data(data, scaler):
    data_close = data[['Close']].values.reshape(-1, 1)
    scaled_data = scaler.transform(data_close)
    x_input = []
    for i in range(100, scaled_data.shape[0]):
        x_input.append(scaled_data[i-100:i])
    x_input = np.array(x_input)
    return x_input

@app.post("/predict")
async def predict(ticker_data: TickerData):
    data = load_data(ticker_data.ticker)
    scaler = MinMaxScaler(feature_range=(0, 1))
    full_data = data[['Close']].values
    scaler.fit(full_data)

    past_100_days = data[['Close']].values[-100:]
    past_100_days = past_100_days.reshape(-1, 1)
    full_data = np.concatenate((past_100_days, data[['Close']].values), axis=0)
    scaled_data = scaler.transform(full_data)

    x_test = []
    for i in range(100, scaled_data.shape[0]):
        x_test.append(scaled_data[i-100:i])
    x_test = np.array(x_test)

    predictions = model.predict(x_test)
    predictions_rescaled = scaler.inverse_transform(predictions)
    actual_values = data['Close'].values[100:]

    return JSONResponse(content={"prediction": predictions_rescaled.tolist(), "actual": actual_values.tolist()})

# Load the pre-trained model
model = load_model('keras_model.h5')

# Streamlit setup
def run_streamlit():
    st.title('Stock Price Prediction')

    # User input for stock ticker symbol
    st.header('Select Stock Ticker Symbol')
    ticker = st.selectbox('Select stock ticker symbol:', list(STOCK_TICKERS.keys()))

    # Load data button
    if st.button('Predict'):
        if ticker:
            try:
                # Fetch data using yfinance
                data = yf.download(ticker, START, TODAY)
                if not data.empty:
                    st.success('Data loaded successfully!')
                    st.write(data.head())  # Display the first few rows of the loaded data
                    
                    # Prepare data for prediction
                    response = requests.post("http://127.0.0.1:8000/predict", json={"ticker": ticker})
                    if response.status_code == 200:
                        prediction_data = response.json()
                        predictions = prediction_data["prediction"]
                        actual = prediction_data["actual"]
                        
                        st.success('Prediction successful!')

                        # Show stock name
                        stock_name = STOCK_TICKERS[ticker]
                        st.subheader(f'Stock: {stock_name} ({ticker})')
                        
                        # Show predicted stock price
                        if predictions:
                            latest_predicted_price = predictions[-1][0]  # The latest predicted price is the last element in the prediction list
                            st.write(f'Predicted Closing Price: ${latest_predicted_price:.2f}')
                        
                        # Plot actual vs predicted prices
                        plt.figure(figsize=(12, 6))
                        plt.plot(actual, label='Actual Price')
                        plt.plot(range(100, 100 + len(predictions)), [p[0] for p in predictions], label='Predicted Price')
                        plt.xlabel('Time')
                        plt.ylabel('Price')
                        plt.legend()
                        plt.grid(True)
                        st.pyplot(plt)
                    else:
                        st.error('Error in prediction API call.')
                else:
                    st.error('No data fetched for the ticker symbol.')
            except Exception as e:
                st.error(f'Error loading data: {e}')
        else:
            st.error('Please select a stock ticker symbol.')

if __name__ == "__main__":
    # Start FastAPI server in a thread
    def run_fastapi():
        uvicorn.run(app, host="127.0.0.1", port=8000)

    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.start()

    # Run Streamlit app
    run_streamlit()
