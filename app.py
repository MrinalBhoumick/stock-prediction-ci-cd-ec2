import os
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import streamlit as st
import threading
import uvicorn
import requests
import subprocess
import json

# Constants
START = "2009-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Predefined list of stock tickers with their names
STOCK_TICKERS = {
    'TCS.NS': 'TCS India',
    'RELIANCE.NS': 'Reliance Industries',
    # Add more tickers as needed
}

# FastAPI setup
app = FastAPI()

class TickerData(BaseModel):
    ticker: str

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

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
        x_test.append(scaled_data[i-100:i, 0])
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
    st.header('Select or Enter Stock Ticker Symbol')
    user_ticker = st.selectbox('Select stock ticker symbol:', list(STOCK_TICKERS.keys()), index=0)
    
    ticker = user_ticker

    # Load data button
    if st.button('Predict'):
        if ticker:
            try:
                # Fetch data using yfinance
                data = yf.download(ticker, START, TODAY)
                if not data.empty:
                    st.success('Data loaded successfully!')
                    st.write(data.head())  # Display the first few rows of the loaded data
                    
                    # Calculate moving averages
                    data['MA100'] = data['Close'].rolling(100).mean()
                    data['MA200'] = data['Close'].rolling(200).mean()

                    # Plot actual stock price and moving averages
                    plt.figure(figsize=(12, 6))
                    plt.plot(data['Close'], label='Actual Price')
                    plt.plot(data['MA100'], label='100-day MA')
                    plt.plot(data['MA200'], label='200-day MA')
                    plt.xlabel('Date')
                    plt.ylabel('Price')
                    plt.legend()
                    plt.title('Stock Price and Moving Averages')
                    plt.grid(True)
                    st.pyplot(plt)

                    # Prepare data for prediction
                    response = requests.post("http://127.0.0.1:8000/predict", json={"ticker": ticker})
                    if response.status_code == 200:
                        prediction_data = response.json()
                        predictions = prediction_data["prediction"]
                        actual = prediction_data["actual"]
                        
                        st.success('Prediction successful!')

                        # Show stock name
                        stock_name = STOCK_TICKERS.get(ticker, ticker)
                        st.subheader(f'Stock: {stock_name} ({ticker})')

                        # Determine if the stock is Indian or foreign
                        if ticker.endswith('.NS'):
                            currency = 'INR'
                        else:
                            currency = 'USD'
                        
                        # Show predicted stock price
                        if predictions:
                            latest_predicted_price = predictions[-1][0]  # The latest predicted price is the last element in the prediction list
                            if currency == 'INR':
                                st.write(f'Predicted Closing Price: ₹{latest_predicted_price:.2f}')
                            else:
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

                    # Run k6 test and display results
                    if st.button('Run Performance Test'):
                        run_k6_test()
                        if os.path.exists('k6_results.json'):
                            with open('k6_results.json', 'r') as file:
                                k6_results = json.load(file)
                                visualize_k6_results(k6_results)
                        else:
                            st.error('Failed to generate k6 test results.')
                else:
                    st.error('No data fetched for the ticker symbol.')
            except Exception as e:
                st.error(f'Error loading data: {e}')
        else:
            st.error('Please enter a stock ticker symbol.')

def run_k6_test():
    try:
        result = subprocess.run(['k6', 'run', 'k6_test.js', '--out', 'json=k6_results.json'], check=True, capture_output=True, text=True)
        st.text(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error(f'Error running k6 test: {e}')

def visualize_k6_results(results):
    # Example of visualizing K6 test results
    metrics = results.get('metrics', {})
    
    if 'http_req_duration' in metrics:
        durations = metrics['http_req_duration']
        time_points = list(range(len(durations)))
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_points, durations, label='Request Duration')
        plt.xlabel('Time')
        plt.ylabel('Duration (ms)')
        plt.title('Request Duration Over Time')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
        
    if 'http_reqs' in metrics:
        total_requests = metrics['http_reqs'].get('total', 0)
        failed_requests = metrics['http_reqs'].get('failed', 0)
        
        st.write(f'Total Requests: {total_requests}')
        st.write(f'Failed Requests: {failed_requests}')
        
        plt.figure(figsize=(12, 6))
        plt.bar(['Total Requests', 'Failed Requests'], [total_requests, failed_requests])
        plt.ylabel('Count')
        plt.title('Request Counts')
        plt.grid(True)
        st.pyplot(plt)

if __name__ == "__main__":
    # Start FastAPI server in a thread
    def run_fastapi():
        uvicorn.run(app, host="127.0.0.1", port=8000)

    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.start()

    # Run Streamlit app
    run_streamlit()
