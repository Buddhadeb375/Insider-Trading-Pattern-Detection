# 📊 Insider Trading Detection using Machine Learning

## 🚀 Project Overview

This project detects **suspicious stock trading activities** using Machine Learning techniques. It combines **price behavior, trading volume, market trends, and news analysis** to identify unusual patterns that may indicate insider trading.

---

## 🎯 Key Features

* 📈 Stock data collection using Yahoo Finance
* 🔍 Anomaly detection using Isolation Forest
* 📊 Market trend filtering (S&P 500 index)
* 📰 News-based validation using NewsAPI
* 📉 Visualization of suspicious trading days
* 📌 KPI metrics for performance analysis

---

## 🛠️ Technologies Used

* Python
* Pandas & NumPy
* Scikit-learn (Isolation Forest)
* Matplotlib
* yFinance API
* NewsAPI

---

## ⚙️ How It Works

### 1. Data Collection

* Fetches stock data (AAPL) and market index (S&P 500)

### 2. Feature Engineering

* Price change percentage
* Volume change percentage
* Market movement

### 3. Anomaly Detection

* Uses **Isolation Forest** to detect unusual trading patterns

### 4. Market Filtering

* Removes anomalies caused by overall market movement

### 5. News Validation

* Checks if anomalies are justified by real-world news

### 6. Visualization

* Highlights suspicious trading days on a stock price graph

---

## 📦 Installation

Install required libraries:

```bash
pip install yfinance pandas numpy scikit-learn matplotlib requests
```

---

## ▶️ Usage

Run the project:

```bash
python stock_anomaly_project.py
```

---

## 🔑 API Configuration

Replace your News API key in the code:

```python
NEWS_API_KEY = "your_api_key_here"
```

Get a free key from: https://newsapi.org

---

## 📊 Sample Output

* KPI metrics (Average price, volume, suspicious days)
* List of suspicious trading dates
* Graph showing anomalies

---

## ⚠️ Limitations

* News API rate limits may slow execution
* Model is unsupervised (no labeled insider trading data)
* Results are indicative, not definitive

---

## 🚀 Future Improvements

* Real-time anomaly detection
* Dashboard using Streamlit
* Integration with live trading alerts
* Advanced NLP for news sentiment analysis

---
