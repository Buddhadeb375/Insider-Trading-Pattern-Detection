import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import requests
import time

# CONFIGURATION
ticker = "AAPL"
market_index = "^GSPC"
start_date = "2024-01-01"
end_date = "2025-10-01"

NEWS_API_KEY = "your_api_key_here"

# DOWNLOAD DATA
print("Downloading data...")

stock = yf.download(ticker, start=start_date, end=end_date)
stock.reset_index(inplace=True)
# Flatten multi-level columns for stock
stock.columns = [col[0] if isinstance(col, tuple) else col for col in stock.columns]
stock.drop(columns='Adj Close', errors='ignore', inplace=True) # Drop 'Adj Close' as 'Close' is used

market = yf.download(market_index, start=start_date, end=end_date)
market.reset_index(inplace=True)
# Flatten multi-level columns for market
market.columns = [col[0] if isinstance(col, tuple) else col for col in market.columns]
market.drop(columns='Adj Close', errors='ignore', inplace=True) # Drop 'Adj Close' as 'Close' is used


# FEATURE ENGINEERING
stock['Price_Change'] = stock['Close'].pct_change()
stock['Volume_Change'] = stock['Volume'].pct_change()
market['Market_Change'] = market['Close'].pct_change()

# Merge datasets
data = pd.merge(stock, market[['Date', 'Market_Change']], on='Date', how='inner')

# Clean data
data.fillna(0, inplace=True)
data.replace([np.inf, -np.inf], 0, inplace=True)

print("Data prepared successfully!")

# ANOMALY DETECTION
features = data[['Price_Change', 'Volume_Change']]

model = IsolationForest(contamination=0.03, random_state=42)
data['Anomaly'] = model.fit_predict(features)

data['Anomaly'] = data['Anomaly'].map({1: 'Normal', -1: 'Suspicious'})

print("Anomaly detection completed!")

# MARKET FILTER
threshold = 0.015
data.loc[(abs(data['Market_Change']) > threshold), 'Anomaly'] = 'Normal'

print("Market filter applied!")

# NEWS CHECK FUNCTION
def has_recent_news(company, date):
    url = f"https://newsapi.org/v2/everything?q={company}&from={date}&to={date}&sortBy=popularity&language=en&apiKey={NEWS_API_KEY}"

    try:
        response = requests.get(url, timeout=5)
        news = response.json()

        if news.get("status") == "ok" and len(news.get("articles", [])) > 0:
            return True

    except Exception as e:
        print(f"News API error on {date}: {e}")

    return False


# NEWS FILTER
print("Checking news... (this may take time)")

for i in range(len(data)):
    if data.loc[i, 'Anomaly'] == 'Suspicious':
        d = data.loc[i, 'Date'].strftime("%Y-%m-%d")

        if has_recent_news(ticker, d):
            data.loc[i, 'Anomaly'] = 'Normal'

        # Prevent API rate limit
        time.sleep(1)

print("News filtering completed!")

# KPI CALCULATIONS
average_close = data['Close'].mean()
max_close = data['Close'].max()
min_close = data['Close'].min()
average_volume = data['Volume'].mean()
suspicious_count = len(data[data['Anomaly'] == 'Suspicious'])

print("\n📊 KPI RESULTS")
print(f"Average Close Price : ${average_close:.2f}")
print(f"Max Close Price     : ${max_close:.2f}")
print(f"Min Close Price     : ${min_close:.2f}")
print(f"Average Volume      : {average_volume:.0f}")
print(f"Suspicious Days     : {suspicious_count}")

# SHOW SUSPICIOUS DAYS
suspicious_days = data[data['Anomaly'] == 'Suspicious']

print("\n⚠ Suspicious Trading Days:")
print(suspicious_days[['Date', 'Close', 'Volume', 'Price_Change', 'Volume_Change']].tail(10))

# PLOT RESULTS
plt.figure(figsize=(12, 6))

plt.plot(data['Date'], data['Close'], label='Stock Price')

plt.scatter(
    suspicious_days['Date'],
    suspicious_days['Close'],
    color='red',
    label='Suspicious Days'
)

plt.title(f"{ticker} - Insider Trading Detection")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend()

plt.show()