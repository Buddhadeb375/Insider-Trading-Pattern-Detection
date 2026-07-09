# Insider Trading Anomaly Detection Dashboard

This is an interactive Streamlit application to download stock and market data, run anomaly detection using Isolation Forest, filter results based on market movements, and cross-reference flags with recent news articles.

## Setup Instructions

We recommend using `uv` to manage python dependencies and run the application.

1. **Install Dependencies and Run**:
   ```bash
   uv run streamlit run app.py
   ```
   Or using standard `pip`:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Streamlit**:
   ```bash
   streamlit run app.py
   ```

## Features
- **Custom Search parameters:** Check any stock ticker against any index (e.g. SPY, S&P 500 indices).
- **Isolation Forest Tuner:** Fine-tune contamination and filter threshold dynamically.
- **News API integration:** Toggle checks and download detailed CSV reports of suspicious days.
