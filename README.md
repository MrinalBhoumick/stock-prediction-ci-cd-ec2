## Stock Price Prediction App using LSTM, FastAPI, and Streamlit

This project demonstrates how to predict stock prices using LSTM (Long Short-Term Memory) neural networks. The prediction model is integrated with FastAPI for creating a backend API and Streamlit for building an interactive web application frontend.

## Features

- **Fetch Data**: Retrieve historical stock price data from Yahoo Finance.
- **Preprocess Data**: Scale and prepare data for LSTM model training.
- **Train LSTM Model**: Build and train an LSTM model using TensorFlow/Keras.
- **Predict Prices**: Make predictions for future stock prices.
- **Interactive UI**: Streamlit interface for users to select stocks, view predictions, and compare with actual prices.
- **Concurrent Execution**: FastAPI serves predictions while Streamlit provides interactive visualization, running concurrently using threading.

## Technologies Used

- Python
- TensorFlow/Keras
- FastAPI
- Streamlit
- pandas, numpy, matplotlib
- yfinance
- scikit-learn (sklearn)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/stock-price-prediction-app.git
   cd stock-price-prediction-app
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   # Create a virtual environment (Linux/macOS)
   python3 -m venv venv
   
   # Activate the virtual environment (Linux/macOS)
   source venv/bin/activate
   
   # Create a virtual environment (Windows)
   python -m venv venv
   
   # Activate the virtual environment (Windows)
   venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. Start the FastAPI server:

   ```bash
   uvicorn app:app --reload
   ```

   This will start the FastAPI server on `http://localhost:8000`.

2. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

   This will launch the Streamlit app in your default web browser.

3. Select a stock ticker from the dropdown menu in the Streamlit app and click on "Predict" to see the predictions and actual prices.

## Files and Directory Structure

- **app.py**: Contains FastAPI backend code for data fetching, model prediction, and API endpoints.
- **LSTM-Model.ipynb**: Jupyter Notebook for LSTM model training and experimentation.
- **README.md**: Project overview, installation guide, and usage instructions.
- **requirements.txt**: List of Python dependencies required to run the application.
- **templates/**: Directory for HTML templates (if any).
- **static/**: Directory for static files (if any).

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your improvements or suggestions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Feel free to adjust the instructions or add more details specific to your project as needed. This setup ensures that users can create a clean environment with all dependencies installed before running the application.
