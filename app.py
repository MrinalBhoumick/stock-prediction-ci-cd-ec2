import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import streamlit as st
import threading
import uvicorn
import io
import base64
import requests
import json

# Constants
START = "2009-01-01"
TODAY = pd.Timestamp.now().strftime("%Y-%m-%d")

# Predefined list of stock tickers with their names
STOCK_TICKERS = {
    'TCS.NS': 'TCS India',
    'RELIANCE.NS': 'Reliance Industries',
    'INFY.NS': 'Infosys Ltd',
    'HDFC.NS': 'HDFC Bank',
    'ICICIBANK.NS': 'ICICI Bank',
    'SBIN.NS': 'State Bank of India',
    'HINDUNILVR.NS': 'Hindustan Unilever',
    'AXISBANK.NS': 'Axis Bank',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank',
    'LT.NS': 'Larsen & Toubro',
    'BAJFINANCE.NS': 'Bajaj Finance',
    'BHARTIARTL.NS': 'Bharti Airtel',
    'ITC.NS': 'ITC Limited',
    'ASIANPAINT.NS': 'Asian Paints',
    'MARUTI.NS': 'Maruti Suzuki',
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

@app.post("/predict")
async def predict(ticker_data: TickerData):
    try:
        data = load_data(ticker_data.ticker)
        
        if data.empty:
            return JSONResponse(content={"error": "No data fetched for the ticker symbol."})

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

        # Ensure y_test matches the length of x_test
        y_test = np.array(data['Close'][100:])
        if len(y_test) < len(x_test):
            y_test = np.pad(y_test, (0, len(x_test) - len(y_test)), mode='constant', constant_values=np.nan)
        y_test = y_test[:x_test.shape[0]]

        if x_test.shape[0] != len(y_test):
            raise ValueError("Mismatch in length between x_test and y_test")

        model = LinearRegression()
        model.fit(x_test, y_test)

        predictions = model.predict(x_test)
        predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))
        actual_values = data['Close'].values[100:]

        # Plot actual vs predicted prices
        plt.figure(figsize=(12, 6))
        plt.plot(actual_values, label='Actual Price')
        plt.plot(range(100, 100 + len(predictions)), predictions_rescaled, label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        
        return JSONResponse(content={"prediction": predictions_rescaled.tolist(), "actual": actual_values.tolist(), "plot_url": plot_url})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

@app.get("/k6_results")
async def get_k6_results():
    try:
        with open('k6_results.json', 'r') as file:
            results = json.load(file)
        return JSONResponse(content={"results": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

# Streamlit setup
def run_streamlit():
    st.title('Stock Price Prediction')

    st.header('Select or Enter Stock Ticker Symbol')
    user_ticker = st.selectbox('Select stock ticker symbol:', list(STOCK_TICKERS.keys()), index=0)
    
    ticker = user_ticker

    if st.button('Predict'):
        if ticker:
            try:
                data = yf.download(ticker, START, TODAY)
                if not data.empty:
                    st.success('Data loaded successfully!')
                    st.write(data.head())  # Display the first few rows of the loaded data
                    
                    data['MA100'] = data['Close'].rolling(100).mean()
                    data['MA200'] = data['Close'].rolling(200).mean()

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

                    response = requests.post("http://127.0.0.1:8000/predict", json={"ticker": ticker})
                    if response.status_code == 200:
                        prediction_data = response.json()
                        if "error" in prediction_data:
                            st.error(f'Error: {prediction_data["error"]}')
                        else:
                            predictions = prediction_data["prediction"]
                            actual = prediction_data["actual"]
                            plot_url = prediction_data["plot_url"]
                            
                            st.success('Prediction successful!')
                            stock_name = STOCK_TICKERS.get(ticker, ticker)
                            st.subheader(f'Stock: {stock_name} ({ticker})')

                            currency = 'INR' if ticker.endswith('.NS') else 'USD'
                            if predictions:
                                latest_predicted_price = predictions[-1][0]
                                st.write(f'Predicted Closing Price: ₹{latest_predicted_price:.2f}' if currency == 'INR' else f'Predicted Closing Price: ${latest_predicted_price:.2f}')
                            
                            st.image(f"data:image/png;base64,{plot_url}", caption='Actual vs Predicted Prices')
                    else:
                        st.error('Error in prediction API call.')
                else:
                    st.error('No data fetched for the ticker symbol.')
            except Exception as e:
                st.error(f'Error loading data: {e}')
        else:
            st.error('Please enter a stock ticker symbol.')

    st.header('K6 Test Results')
    try:
        k6_results = requests.get("http://127.0.0.1:8000/k6_results")
        if k6_results.status_code == 200:
            results = k6_results.json()
            if "results" in results:
                st.write(results["results"])
            else:
                st.error(f'Error: {results["error"]}')
        else:
            st.error('Error fetching k6 results.')
    except Exception as e:
        st.error(f'Error fetching k6 results: {e}')

if __name__ == "__main__":
    def run_fastapi():
        uvicorn.run(app, host="127.0.0.1", port=8000)

    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.start()

    run_streamlit()
