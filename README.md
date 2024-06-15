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

pandas: Used for data manipulation and handling.
datetime: Handles dates and times.
matplotlib.pyplot: Provides plotting functions.
yfinance: Fetches financial data from Yahoo Finance.
numpy: Essential for numerical operations.
tensorflow.keras: Implements deep learning models like LSTM.
sklearn.preprocessing.MinMaxScaler: Scales data between 0 and 1.
fastapi: Framework for building APIs quickly with Python.
pydantic.BaseModel: Used for data validation in FastAPI.
streamlit: For building interactive web applications.
threading: Enables concurrent execution of FastAPI and Streamlit.
uvicorn: ASGI server for running FastAPI applications.
requests: Makes HTTP requests to the FastAPI backend.

=================================================================================================================================

START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Predefined list of stock tickers with their names
STOCK_TICKERS = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    # Add more stock tickers as needed
}

START and TODAY: Constants for specifying the start and end dates for fetching stock data.
STOCK_TICKERS: A dictionary mapping stock ticker symbols to their corresponding company names. This will be used to populate the dropdown menu in the Streamlit app.

=================================================================================================================================

app = FastAPI()

class TickerData(BaseModel):
    ticker: str


=================================================================================================================================

app: Instance of FastAPI used to define API endpoints.
TickerData: Pydantic BaseModel for validating JSON data sent to the /predict endpoint. It ensures that the request body contains a ticker field of type str.

=================================================================================================================================

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

===============================================================================================================================

load_data: Function that uses Yahoo Finance (yf.download) to fetch historical stock data for a given ticker from START to TODAY.

================================================================================================================================

def preprocess_data(data, scaler):
    data_close = data[['Close']].values.reshape(-1, 1)
    scaled_data = scaler.transform(data_close)
    x_input = []
    for i in range(100, scaled_data.shape[0]):
        x_input.append(scaled_data[i-100:i])
    x_input = np.array(x_input)
    return x_input


================================================================================================================================

preprocess_data: Function that preprocesses the fetched data (data) by scaling the closing prices ('Close' column) and creating input sequences (x_input) for the LSTM model.


================================================================================================================================

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

@app.post("/predict"): Decorator for defining a POST endpoint /predict in FastAPI.
predict: Function that handles the /predict endpoint. It loads data for the specified ticker, preprocesses it, makes predictions using a pre-trained LSTM model (model), and returns JSON response containing predicted (predictions_rescaled) and actual (actual_values) closing prices.

========================================================================================================================================


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


====================================================================================================================================

run_streamlit: Function that sets up the Streamlit app interface. It displays a dropdown menu to select a stock ticker symbol (ticker). When the user clicks "Predict," it fetches data for the selected ticker, sends a request to the FastAPI /predict endpoint, retrieves predictions, and displays them along with actual prices in a plot.




if __name__ == "__main__":
    # Start FastAPI server in a thread
    def run_fastapi():
        uvicorn.run(app, host="127.0.0.1", port=8000)

    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.start()

    # Run Streamlit app
    run_streamlit()



if name == "main": Checks if the script is being run directly.
run_fastapi: Function that starts the FastAPI server (uvicorn.run) in a separate thread (fastapi_thread).
run_streamlit: Function call to start the Streamlit application, allowing both FastAPI and Streamlit to run concurrently.
Explanation Summary
This app.py script integrates FastAPI and Streamlit for stock price prediction using an LSTM model. It fetches historical stock data, preprocesses it, makes predictions, and visualizes actual vs predicted prices. FastAPI serves as the backend API, handling data fetching, processing, and prediction, while Streamlit provides the frontend interface for user interaction and visualization. The two are coordinated to run concurrently using threading.





