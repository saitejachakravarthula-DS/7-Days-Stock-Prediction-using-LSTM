### Predicting the next 7 values 




# Import necessary libraries
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime

# Load the trained LSTM model and define the scaler
model = load_model('/workspaces/Stock-Prediction-using-LSTM/Latest_stock_price_model.LSTM.keras')  
scaler = MinMaxScaler(feature_range=(0, 1))

# Set up Streamlit app layout
st.title("Stock Price Prediction App")
st.write("This app predicts the next 7 days of stock prices for a selected stock.")

# Stock symbol input
stock_symbol = st.text_input("Enter Stock Symbol (e.g., GOOG for Google):", "GOOG")
start = datetime.now() - pd.DateOffset(years=1)  # 1 year of historical data
end = datetime.now()

# Step 1: Fetch the latest stock data
@st.cache_data
def fetch_data(stock_symbol):
    stock_data = yf.download(stock_symbol, start=start, end=end)
    return stock_data[['Adj Close']]

adj_close_data = fetch_data(stock_symbol)

# Step 2: Scale the adjusted close data
scaler.fit(adj_close_data)  # Fit the scaler on the full data range
scaled_data = scaler.transform(adj_close_data)  # Scale the data

# Step 3: Extract the last 100 days of scaled adjusted close prices for prediction
last_100_days_scaled = scaled_data[-100:]

# Define the function for predicting the next 7 days
def predict_next_week(last_100_days_scaled, model, scaler):
    predictions = []
    input_sequence = last_100_days_scaled.reshape(1, -1, 1)

    for _ in range(7):  # Predicting for 7 days
        pred_price = model.predict(input_sequence)
        predictions.append(pred_price[0, 0])

        # Ensure pred_price has the correct shape and update input_sequence
        pred_price_reshaped = np.reshape(pred_price, (1, 1, 1))
        input_sequence = np.append(input_sequence[:, 1:, :], pred_price_reshaped, axis=1)

    # Inverse transform to get actual prices
    predictions_scaled = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions_scaled)

# Run the prediction
predicted_prices = predict_next_week(last_100_days_scaled, model, scaler)

# Display the predicted prices in a table
future_dates = pd.date_range(start=end + pd.DateOffset(days=1), periods=7, freq='B')  # Only business days
predicted_df = pd.DataFrame(predicted_prices, index=future_dates, columns=["Predicted Price"])
st.write("Predicted prices for the next 7 days:")
st.table(predicted_df)

# Plot the predictions along with the last 100 days of actual data
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(np.arange(100), adj_close_data['Adj Close'][-100:].values, label="Last 100 Days (Actual)")
ax.plot(np.arange(100, 107), predicted_prices, marker='o', linestyle='-', color='red', label="Predictions (Next 7 Days)")
ax.set_xlabel("Days")
ax.set_ylabel("Stock Price")
ax.set_title(f"Stock Price Prediction for {stock_symbol} - Next 7 Days")
ax.legend()

# Display the plot
st.pyplot(fig)
