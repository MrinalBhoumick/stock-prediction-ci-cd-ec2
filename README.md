

# Stock Price Prediction using LSTM, FastAPI, and Streamlit

This repository contains a stock price prediction application that uses a Long Short-Term Memory (LSTM) neural network to predict future stock prices. The application is built using FastAPI and Streamlit, and can be deployed using Docker.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
  - [Creating a Virtual Environment](#creating-a-virtual-environment)
  - [Running the Application](#running-the-application)
  - [Docker Setup](#docker-setup)
- [Usage](#usage)
- [How It Works](#how-it-works)
  - [LSTM Overview](#lstm-overview)
  - [Benefits of LSTM](#benefits-of-lstm)
  - [Other Algorithms](#other-algorithms)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This application allows users to predict stock prices for various companies using historical stock data. The predictions are made using an LSTM neural network, which is particularly suited for time series forecasting.

## Features

- Predict stock prices using an LSTM neural network.
- View historical stock prices and predicted future prices.
- Interactive user interface built with Streamlit.
- API for fetching predictions built with FastAPI.

## Requirements

- Python 3.7 or higher
- Docker (optional, for containerized deployment)

## Setup

### Creating a Virtual Environment

To set up the application in a virtual environment, follow these steps:

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/stock-price-prediction
    cd stock-price-prediction
    ```

2. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

### Running the Application

1. Start the FastAPI server:

    ```sh
    uvicorn app:app --reload
    ```

2. In a new terminal, run the Streamlit app:

    ```sh
    streamlit run app.py
    ```

### Docker Setup

To build and run the application using Docker:

1. Build the Docker image:

    ```sh
    docker build -t stock-price-prediction-app .
    ```

2. Run the Docker container:

    ```sh
    docker run -d -p 8000:8000 -p 8501:8501 stock-price-prediction-app
    ```

## Usage

1. Open your web browser and navigate to `http://localhost:8501` to access the Streamlit interface.
2. Select a stock ticker symbol from the dropdown menu and click "Predict" to see the predictions.

## How It Works

### LSTM Overview

LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) that is well-suited for sequence prediction problems. It can learn and remember over long sequences and is resistant to the vanishing gradient problem.

### Benefits of LSTM

- **Handles Long-Term Dependencies**: LSTMs can capture long-term dependencies in the data.
- **Resistant to Vanishing Gradient**: The architecture of LSTMs helps mitigate the vanishing gradient problem, making them suitable for long time series.

### Other Algorithms

While LSTMs are highly effective for time series prediction, other algorithms like ARIMA, Prophet, and simple feedforward neural networks can also be used. However, LSTMs generally provide better performance for complex patterns in time series data.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---