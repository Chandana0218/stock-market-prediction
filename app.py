import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime

# ----------------------------
# Set Dark Plot Style
# ----------------------------
plt.style.use("dark_background")

# Set app title
st.title("📈 Stock Price Predictor & Future Forecaster")

# Sidebar input
st.sidebar.header("Stock Input")
stock = st.sidebar.text_input("Enter Stock Ticker (e.g., GOOG, AAPL, MSFT)", "")

# Only run logic when ticker is entered
if stock:
    # Set date range: last 20 years
    end = datetime.now()
    start = datetime(end.year - 20, end.month, end.day)

    @st.cache_data
    def get_data(ticker):
        try:
            data = yf.download(ticker, start=start, end=end)
            if data.empty:
                return None
            return data
        except:
            return None

    google_data = get_data(stock)

    if google_data is None:
        st.error(f"❌ Failed to retrieve data for: {stock}")
        st.stop()

    # Load model
    try:
        model = load_model("Latest_stock_price_model.keras")
    except:
        st.error("⚠️ Error loading model. Make sure 'Latest_stock_price_model.keras' is present.")
        st.stop()

    st.subheader(f"Showing data for: {stock}")
    st.dataframe(google_data.tail())

    # ----------------------------
    # Moving Averages
    # ----------------------------
    st.subheader(" Moving Averages")
    for days in [100, 200, 250]:
        ma_col = f"MA_{days}"
        google_data[ma_col] = google_data['Close'].rolling(window=days).mean()
        fig = plt.figure(figsize=(14, 6))
        plt.plot(google_data['Close'], label="Close Price", color='cyan')
        plt.plot(google_data[ma_col], label=f"{days}-Day MA", linestyle='--', color='orange')
        plt.legend()
        plt.title(f"{stock} Close Price & {days}-Day MA")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(alpha=0.3)
        st.pyplot(fig)

    # MA 100 vs MA 250 Comparison
    st.subheader(" MA 100 vs MA 250 Comparison")
    fig = plt.figure(figsize=(14, 6))
    plt.plot(google_data['MA_100'], label="MA 100", color='lime')
    plt.plot(google_data['MA_250'], label="MA 250", color='magenta')
    plt.plot(google_data['Close'], label="Close", alpha=0.4, color='white')
    plt.legend()
    plt.title("MA 100 vs MA 250 Comparison")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(alpha=0.3)
    st.pyplot(fig)

    # ----------------------------
    # Prepare test data
    # ----------------------------
    splitting_len = int(len(google_data) * 0.7)
    x_test = google_data[['Close']].iloc[splitting_len:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test[['Close']])

    x_data = []
    y_data = []

    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i - 100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    # Predictions
    predictions = model.predict(x_data)
    inv_predictions = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    # Show prediction results
    plot_data = pd.DataFrame({
        'Original': inv_y_test.flatten(),
        'Predicted': inv_predictions.flatten()
    }, index=google_data.index[splitting_len + 100:])

    st.subheader(" Prediction Results (Table)")
    st.dataframe(plot_data.tail())

    st.subheader(" Original vs Predicted Plot")
    fig2 = plt.figure(figsize=(14, 6))
    plt.plot(plot_data['Original'], label='Original Price', color='cyan')
    plt.plot(plot_data['Predicted'], label='Predicted Price', color='red')
    plt.legend()
    plt.title("Actual vs Predicted Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(alpha=0.3)
    st.pyplot(fig2)

    # ----------------------------
    # Model Performance
    # ----------------------------
    st.subheader(" Can I Use This Model for Future Predictions?")

    # RMSE Evaluation
    rmse = np.sqrt(mean_squared_error(inv_y_test, inv_predictions))
    st.write(f"**Model RMSE on Test Data:** `{rmse:.2f}`")

    if rmse < 10:
        st.success("✅ High accuracy. Safe for trend-based investments.")
    elif rmse < 30:
        st.info("⚠️ Moderate accuracy. Cross-check with other tools.")
    else:
        st.error("❌ High error. Not reliable for investment decisions.")

    # ----------------------------
    # Future Price Forecasting
    # ----------------------------
    st.subheader(" Forecast Future Prices")

    future_days = st.slider("Select number of future days to forecast", min_value=1, max_value=30, value=7)

    last_100_days = scaled_data[-100:]
    future_input = list(last_100_days)
    future_predictions = []

    for _ in range(future_days):
        input_array = np.array(future_input[-100:])
        input_array = input_array.reshape(1, 100, 1)
        pred = model.predict(input_array, verbose=0)
        future_predictions.append(pred[0, 0])
        future_input.append(pred[0])

    future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    last_date = google_data.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days, freq='B')

    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted Price': future_prices.flatten()})
    forecast_df.set_index('Date', inplace=True)

    st.subheader(" Forecasted Prices Table")
    st.dataframe(forecast_df)

    st.subheader(" Forecasted Price Plot")
    fig3 = plt.figure(figsize=(14, 6))
    plt.plot(google_data['Close'].tail(100), label='Last 100 Days (Actual)', color='cyan')
    plt.plot(forecast_df['Forecasted Price'], label='Forecasted Prices', color='lime', linestyle='--')
    plt.legend()
    plt.title(f"{stock} Forecast for Next {future_days} Days")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(alpha=0.3)
    st.pyplot(fig3)
else:
    st.info("👆Please enter a stock ticker in the sidebar to view results.")
