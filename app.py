import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import datetime
import time

# ---------------------------------------------------------
# PAGE CONFIGURATION & THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="Insider Trading Anomaly Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Metric Cards Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #00F2FE;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #A3AED0;
    }
    .metric-card {
        background-color: #111C44;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid #1E293B;
    }
    
    /* Main title layout */
    .title-container {
        padding: 1.5rem 0rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid #1E293B;
    }
    .subtitle {
        color: #A3AED0;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR / CONTROLS
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/nolan/96/combo-chart.png", width=80)
st.sidebar.title("Configuration")
st.sidebar.markdown("Configure stock parameters and Isolation Forest sensitivity.")

# Inputs
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL", help="e.g. AAPL, MSFT, TSLA").upper().strip()
market_index = st.sidebar.text_input("Market Index Ticker", value="^GSPC", help="S&P 500 is ^GSPC, NASDAQ is ^IXIC").strip()

# Date range selection
col_start, col_end = st.sidebar.columns(2)
with col_start:
    start_date = st.date_input("Start Date", datetime.date(2024, 1, 1))
with col_end:
    end_date = st.date_input("End Date", datetime.date(2025, 10, 1))

st.sidebar.subheader("Model Hyperparameters")
contamination = st.sidebar.slider(
    "Contamination (Anomaly %)",
    min_value=0.01,
    max_value=0.15,
    value=0.03,
    step=0.01,
    help="The proportion of outliers in the data set."
)
market_threshold = st.sidebar.slider(
    "Market Filter Threshold",
    min_value=0.005,
    max_value=0.05,
    value=0.015,
    step=0.005,
    help="Daily market return threshold. Anomalies occurring on days with market changes larger than this will be ignored (assumed driven by macro market trends)."
)

st.sidebar.subheader("News Verification")
news_api_key = st.sidebar.text_input(
    "News API Key", 
    type="password", 
    help="Enter your News API key from newsapi.org to filter out days with public news events."
)

use_mock_news = False
if not news_api_key:
    st.sidebar.info("💡 No News API Key provided. You can enable Mock News for demonstration purposes.")
    use_mock_news = st.sidebar.checkbox("Simulate / Mock News Verification", value=True)

# ---------------------------------------------------------
# DATA FETCHING (CACHED)
# ---------------------------------------------------------
@st.cache_data(show_spinner="Downloading market data from Yahoo Finance...")
def fetch_data(ticker, market_index, start, end):
    try:
        # Convert date to string format expected by yfinance
        s_str = start.strftime("%Y-%m-%d")
        e_str = end.strftime("%Y-%m-%d")
        
        stock_df = yf.download(ticker, start=s_str, end=e_str)
        if stock_df.empty:
            return None, f"No data found for stock ticker {ticker}."
            
        stock_df.reset_index(inplace=True)
        # Flatten multi-level columns if present
        stock_df.columns = [col[0] if isinstance(col, tuple) else col for col in stock_df.columns]
        stock_df.drop(columns='Adj Close', errors='ignore', inplace=True)
        
        market_df = yf.download(market_index, start=s_str, end=e_str)
        if market_df.empty:
            return None, f"No data found for market ticker {market_index}."
            
        market_df.reset_index(inplace=True)
        market_df.columns = [col[0] if isinstance(col, tuple) else col for col in market_df.columns]
        market_df.drop(columns='Adj Close', errors='ignore', inplace=True)
        
        return (stock_df, market_df), None
    except Exception as e:
        return None, str(e)

# ---------------------------------------------------------
# NEWS CHECK FUNCTION (CACHED)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def has_recent_news_cached(company, date_str, api_key):
    url = f"https://newsapi.org/v2/everything?q={company}&from={date_str}&to={date_str}&sortBy=popularity&language=en&apiKey={api_key}"
    try:
        response = requests.get(url, timeout=5)
        news = response.json()
        if news.get("status") == "ok" and len(news.get("articles", [])) > 0:
            return True, news.get("articles")[:3] # Return top 3 articles
    except Exception as e:
        pass
    return False, []

def get_mock_news(company, date_str):
    # Deterministic mock based on date hash to avoid flapping on re-runs
    h = hash(date_str + company)
    has_news = (h % 2 == 0)
    if has_news:
        return True, [
            {"title": f"Earnings Report: {company} Beats Q4 Projections", "source": {"name": "MarketWatch"}},
            {"title": f"Analyst upgrades {company} to BUY following product launch", "source": {"name": "Reuters"}},
            {"title": f"Why shares of {company} are rallying today", "source": {"name": "Motley Fool"}}
        ]
    return False, []

# ---------------------------------------------------------
# MAIN APP LAYOUT
# ---------------------------------------------------------
st.markdown(f"""
<div class="title-container">
    <h1 style='margin:0;'>🕵️‍♂️ Insider Trading & Anomaly Detector</h1>
    <div class="subtitle">Detecting suspicious trading volume and price movements for <b>{ticker}</b> relative to index <b>{market_index}</b></div>
</div>
""", unsafe_allow_html=True)

# Fetch Data
data_pack, err = fetch_data(ticker, market_index, start_date, end_date)

if err:
    st.error(f"❌ Error loading data: {err}")
    st.info("Please verify the tickers are correct and try a different date range.")
else:
    stock, market = data_pack
    
    # Feature Engineering
    stock['Price_Change'] = stock['Close'].pct_change()
    stock['Volume_Change'] = stock['Volume'].pct_change()
    market['Market_Change'] = market['Close'].pct_change()

    # Merge
    merged_data = pd.merge(stock, market[['Date', 'Market_Change']], on='Date', how='inner')
    merged_data.fillna(0, inplace=True)
    merged_data.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Run Isolation Forest Anomaly Detection
    features = merged_data[['Price_Change', 'Volume_Change']]
    model = IsolationForest(contamination=contamination, random_state=42)
    merged_data['Anomaly_Raw'] = model.fit_predict(features)
    
    # Map raw predictions (-1: suspicious, 1: normal)
    merged_data['Anomaly'] = merged_data['Anomaly_Raw'].map({1: 'Normal', -1: 'Suspicious'})
    
    # Save a record of the technical anomaly count before filtering
    tech_anomaly_count = len(merged_data[merged_data['Anomaly'] == 'Suspicious'])
    
    # Apply Market Filter
    # (Ignore stock anomalies if the broader market moved heavily in either direction)
    merged_data.loc[(abs(merged_data['Market_Change']) > market_threshold), 'Anomaly'] = 'Normal'
    market_filtered_count = len(merged_data[merged_data['Anomaly'] == 'Suspicious'])
    
    # Apply News Filter
    news_details = {} # date -> list of articles
    news_filtered_out = 0
    
    suspicious_indices = merged_data[merged_data['Anomaly'] == 'Suspicious'].index
    
    if len(suspicious_indices) > 0:
        progress_bar = st.progress(0, text="Performing News API validation...")
        
        for idx, i in enumerate(suspicious_indices):
            d_str = merged_data.loc[i, 'Date'].strftime("%Y-%m-%d")
            
            has_news = False
            articles = []
            
            if news_api_key:
                has_news, articles = has_recent_news_cached(ticker, d_str, news_api_key)
                time.sleep(0.5) # rate limit prevention
            elif use_mock_news:
                has_news, articles = get_mock_news(ticker, d_str)
                
            if has_news:
                merged_data.loc[i, 'Anomaly'] = 'Normal'
                news_filtered_out += 1
                news_details[d_str] = articles
            
            progress_bar.progress((idx + 1) / len(suspicious_indices), text=f"Checked {d_str}...")
        
        progress_bar.empty()

    suspicious_days = merged_data[merged_data['Anomaly'] == 'Suspicious']
    final_suspicious_count = len(suspicious_days)

    # ---------------------------------------------------------
    # DISPLAY METRICS
    # ---------------------------------------------------------
    # Format metrics cleanly
    average_close = merged_data['Close'].mean()
    max_close = merged_data['Close'].max()
    min_close = merged_data['Close'].min()
    average_volume = merged_data['Volume'].mean()

    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric(label="Average Close", value=f"${average_close:.2f}")
    with m_col2:
        st.metric(label="Max Close", value=f"${max_close:.2f}")
    with m_col3:
        st.metric(label="Min Close", value=f"${min_close:.2f}")
    with m_col4:
        st.metric(label="Average Volume", value=f"{average_volume:,.0f}")
    with m_col5:
        st.metric(
            label="Suspicious Days", 
            value=f"{final_suspicious_count}", 
            delta=f"-{tech_anomaly_count - final_suspicious_count} filtered",
            delta_color="normal"
        )
        
    # Anomaly breakdown alert
    if final_suspicious_count > 0:
        st.warning(f"⚠️ **Anomaly Alert:** Found **{final_suspicious_count}** days with highly suspicious volume and price movement that are **not** explained by market movements or news events.")
    else:
        st.success("✅ **No unexplained anomalies found.** All suspicious trading days were explained by market movements or news releases.")

    # ---------------------------------------------------------
    # TABS FOR CHARTS AND LOGS
    # ---------------------------------------------------------
    tab_chart, tab_data, tab_news_logs = st.tabs(["📊 Interactive Chart", "📋 Anomaly Data Log", "📰 News Analysis Log"])
    
    with tab_chart:
        # Plotly Stock Chart
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.08, 
            row_heights=[0.7, 0.3]
        )
        
        # 1. Close Price trace
        fig.add_trace(
            go.Scatter(
                x=merged_data['Date'], 
                y=merged_data['Close'], 
                name="Stock Close Price",
                line=dict(color="#00F2FE", width=2)
            ),
            row=1, col=1
        )
        
        # 2. Suspicious overlay
        fig.add_trace(
            go.Scatter(
                x=suspicious_days['Date'],
                y=suspicious_days['Close'],
                mode='markers',
                name='Suspicious Day',
                marker=dict(
                    color='#FF4B4B',
                    size=12,
                    symbol='circle',
                    line=dict(color='white', width=1.5)
                ),
                hovertemplate=(
                    "<b>Suspicious Trading Day</b><br>" +
                    "Date: %{x|%Y-%m-%d}<br>" +
                    "Close Price: $%{y:.2f}<br>" +
                    "Price Change: %{customdata[0]:+.2%}<br>" +
                    "Volume Change: %{customdata[1]:+.2%}<extra></extra>"
                ),
                customdata=np.stack((suspicious_days['Price_Change'], suspicious_days['Volume_Change']), axis=-1)
            ),
            row=1, col=1
        )
        
        # 3. Volume trace
        # Distinguish volume bars for anomalies
        volume_colors = ['rgba(0, 242, 254, 0.3)'] * len(merged_data)
        for i in suspicious_days.index:
            volume_colors[i] = '#FF4B4B'
            
        fig.add_trace(
            go.Bar(
                x=merged_data['Date'],
                y=merged_data['Volume'],
                name="Trading Volume",
                marker_color=volume_colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Format Layout
        fig.update_layout(
            height=600,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=50, b=40),
            hovermode="x"
        )
        fig.update_yaxes(title_text="Stock Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
    with tab_data:
        st.subheader("Suspicious Trading Log")
        if final_suspicious_count > 0:
            # Clean dataframe for presentation
            log_df = suspicious_days[['Date', 'Close', 'Volume', 'Price_Change', 'Volume_Change']].copy()
            log_df['Date'] = log_df['Date'].dt.strftime("%Y-%m-%d")
            log_df['Price_Change'] = log_df['Price_Change'].map(lambda x: f"{x:+.2%}")
            log_df['Volume_Change'] = log_df['Volume_Change'].map(lambda x: f"{x:+.2%}")
            log_df['Close'] = log_df['Close'].map(lambda x: f"${x:.2f}")
            log_df['Volume'] = log_df['Volume'].map(lambda x: f"{x:,.0f}")
            
            st.dataframe(log_df, use_container_width=True)
            
            # Download button
            csv = suspicious_days.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Anomaly Report as CSV",
                data=csv,
                file_name=f"{ticker}_suspicious_trading_report.csv",
                mime="text/csv"
            )
        else:
            st.info("No suspicious trading days detected in log.")
            
    with tab_news_logs:
        st.subheader("Filtered & Verified News Analysis")
        st.markdown(
            "Here are the suspicious trading days that were automatically **cleared (marked Normal)** "
            "because the tool discovered related public news releases on those dates."
        )
        
        if news_filtered_out > 0:
            for d_str, articles in news_details.items():
                with st.expander(f"📅 Date: {d_str} ({len(articles)} News Event(s) Found)"):
                    for art in articles:
                        title = art.get('title')
                        source = art.get('source', {}).get('name', 'Unknown')
                        url = art.get('url', '#')
                        st.markdown(f"- **{title}** *(Source: {source})*")
                        if url != '#':
                            st.markdown(f"  [Read Full Article]({url})")
        else:
            st.info("No anomalies were cleared by news verification in this run.")

    # ---------------------------------------------------------
    # METHODOLOGY / DESCRIPTION
    # ---------------------------------------------------------
    with st.expander("🔬 Methodology Explanation"):
        st.markdown("""
        ### How the Detection Pipeline Works
        
        1. **Feature Engineering**: 
           - The model extracts the percentage daily change in **Stock Closing Price** and **Trading Volume**. Insider trading is often characterized by large price changes on atypically high volume *before* public announcements.
           
        2. **Isolation Forest Model**:
           - An unsupervised learning algorithm (`Isolation Forest`) isolates outliers (anomalies) in the 2D feature space of (Price Change, Volume Change). 
           - **Contamination** dictates the expected ratio of anomalies in the dataset.
           
        3. **Market Filtering**:
           - If the broader market (e.g. S&P 500) had a major move on the same day (greater than the **Market Filter Threshold**), we ignore the anomaly. This filters out stock moves that were simply driven by macro-economic events or market-wide volatility.
           
        4. **News Event Verification**:
           - For remaining outliers, the pipeline checks the **News API** for press releases or articles about the stock. If positive news (e.g., earnings beats, product launches) or negative news occurred, the anomaly is marked as **Normal** because public news explains the trade volume.
           - Unexplained outliers remain labeled as **Suspicious**.
        """)
