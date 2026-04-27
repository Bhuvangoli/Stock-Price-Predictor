import streamlit as st
import pandas as pd
import datetime
from database import init_db, add_query, get_recent_queries, add_alert, get_active_alerts, deactivate_alert
from data_fetcher import fetch_stock_data, add_technical_indicators, fetch_news_sentiment, get_recommendation
from model_pipeline import load_or_train_dl_model, generate_ensemble_forecast
from ui_components import plot_candlestick, plot_indicators, plot_predictions, plot_forecast, metric_card, display_signal

# --- Page Config ---
st.set_page_config(page_title="Intelligent Stock Platform", layout="wide", page_icon="📈")

# --- Custom CSS (Premium Dark Theme) ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* App background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1321 50%, #0a1628 100%);
        min-height: 100vh;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #112240 100%);
        border-right: 1px solid rgba(100, 200, 255, 0.1);
    }
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stSelectbox select {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(100, 200, 255, 0.2);
        color: white;
        border-radius: 8px;
    }

    /* Main header */
    h1 {
        background: linear-gradient(90deg, #64DFDF, #6930C3, #5390D9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700 !important;
        font-size: 2.4rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Subheaders */
    h2, h3 {
        color: #a8dadc !important;
        font-weight: 600 !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(100, 200, 255, 0.1);
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: rgba(255,255,255,0.6);
        font-weight: 500;
        padding: 8px 20px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(100,223,223,0.2), rgba(105,48,195,0.2)) !important;
        color: #64dfdf !important;
        border-bottom: 2px solid #64dfdf !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6930C3, #5390D9);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(105, 48, 195, 0.3);
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 25px rgba(105, 48, 195, 0.6);
        transform: translateY(-1px);
        background: linear-gradient(135deg, #7B41D5, #64A0E5);
    }
    .stButton > button:active {
        transform: translateY(0px);
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
        border: 1px solid rgba(100, 223, 223, 0.2);
        border-radius: 12px;
        padding: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    [data-testid="metric-container"]:hover {
        border-color: rgba(100, 223, 223, 0.5);
        box-shadow: 0 6px 30px rgba(100, 223, 223, 0.15);
        transform: translateY(-2px);
    }
    [data-testid="metric-container"] label {
        color: rgba(168, 218, 220, 0.8) !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, rgba(83, 144, 217, 0.2), rgba(100, 223, 223, 0.2));
        border: 1px solid rgba(100, 223, 223, 0.4);
        color: #64dfdf;
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
        padding: 0.6rem;
        transition: all 0.3s ease;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, rgba(83, 144, 217, 0.4), rgba(100, 223, 223, 0.4));
        box-shadow: 0 4px 20px rgba(100, 223, 223, 0.25);
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #64dfdf !important;
    }

    /* Alerts / Messages */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #64dfdf;
        background: rgba(100, 223, 223, 0.08);
    }

    /* DataFrame */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(100, 223, 223, 0.15);
    }

    /* Dividers */
    hr {
        border-color: rgba(100, 223, 223, 0.1) !important;
    }

    /* Sidebar title */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #64dfdf !important;
    }

    /* Number input */
    .stNumberInput input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(100, 200, 255, 0.2) !important;
        border-radius: 8px !important;
        color: white !important;
    }

    /* Select box */
    .stSelectbox [data-baseweb="select"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(100, 200, 255, 0.2);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Init DB ---
init_db()

# --- Sidebar ---
st.sidebar.title("⚙️ Configuration")
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper()
period = st.sidebar.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

if st.sidebar.button("Load Data"):
    add_query(ticker)

# --- Main App ---
st.title(f"📈 Intelligent Stock Analysis: {ticker}")

# Fetch Data
with st.spinner(f"Fetching data for {ticker}..."):
    raw_df = fetch_stock_data(ticker, period)

if raw_df is None or raw_df.empty:
    st.error("Failed to load data. Please check the ticker symbol.")
    st.stop()

# Add Indicators
df = add_technical_indicators(raw_df)

# Check Alerts
active_alerts = get_active_alerts()
current_price = df['Close'].iloc[-1]
for alert in active_alerts:
    a_id, a_ticker, target, cond = alert
    if a_ticker == ticker:
        if (cond == 'above' and current_price >= target) or (cond == 'below' and current_price <= target):
            st.success(f"🚨 **ALERT!** {ticker} has crossed {cond} ${target:.2f}! (Current: ${current_price:.2f})")
            deactivate_alert(a_id)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🤖 Predictions", "⚙️ Training", "🔔 Alerts & History"])

# --- Tab 1: Dashboard ---
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Current Price", f"${current_price:.2f}", f"{df['Close'].iloc[-1] - df['Close'].iloc[-2]:.2f}")
    with col2:
        metric_card("Volume", f"{df['Volume'].iloc[-1]:,}")
    with col3:
        metric_card("RSI (14)", f"{df['RSI'].iloc[-1]:.2f}" if not pd.isna(df['RSI'].iloc[-1]) else "N/A")
    with col4:
        st.download_button(
            label="Download Data as CSV",
            data=df.to_csv().encode('utf-8'),
            file_name=f'{ticker}_data.csv',
            mime='text/csv',
        )

    st.plotly_chart(plot_candlestick(df, ticker), width='stretch')
    st.plotly_chart(plot_indicators(df), width='stretch')
    
    # News & Sentiment
    st.subheader("📰 Latest News & Sentiment")
    news_data = fetch_news_sentiment(ticker)
    
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.markdown(f"**Average Sentiment Score:** {news_data['avg_score']:.2f}")
        st.markdown(f"**Overall Sentiment:** {news_data['overall']}")
        rec, signals = get_recommendation(df, news_data['avg_score'])
        st.markdown("### Recommendation")
        display_signal(rec)
        for sig in signals:
            st.write(f"- {sig}")
            
    with col_s2:
        for item in news_data['items'][:5]:
            st.markdown(f"[{item['title']}]({item['link']}) - *{item['sentiment_label']}*")

# --- Tab 2: Predictions ---
with tab2:
    st.subheader("🔮 Model Predictions & Forecasting")
    
    if st.button("Generate Forecasts (Ensemble)"):
        with st.spinner("Loading models and generating forecasts..."):
            ensemble_res = generate_ensemble_forecast(ticker, df)
            
            # Validation Plot
            y_true = ensemble_res['LSTM']['y_true']
            y_preds = {
                'LSTM': ensemble_res['LSTM']['y_pred'],
                'GRU': ensemble_res['GRU']['y_pred']
            }
            st.plotly_chart(plot_predictions(y_true, y_preds), width='stretch')
            
            # Metrics
            st.markdown("### Model Evaluation Metrics")
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.write("**LSTM Metrics**")
                st.json(ensemble_res['LSTM']['metrics'])
            with m_col2:
                st.write("**GRU Metrics**")
                st.json(ensemble_res['GRU']['metrics'])
                
            # Future Forecasts
            st.markdown("### Future Forecasting")
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                st.write("**Next 7 Days**")
                forecast_7 = {
                    'LSTM': ensemble_res['LSTM']['future_7'],
                    'GRU': ensemble_res['GRU']['future_7'],
                    'ARIMA': ensemble_res['ARIMA_7'],
                    'Ensemble': ensemble_res['Ensemble_7']
                }
                st.plotly_chart(plot_forecast(forecast_7, 7), width='stretch')
                st.dataframe(pd.DataFrame(forecast_7))
                
            with f_col2:
                st.write("**Next 30 Days**")
                forecast_30 = {
                    'LSTM': ensemble_res['LSTM']['future_30'],
                    'GRU': ensemble_res['GRU']['future_30'],
                    'ARIMA': ensemble_res['ARIMA_30'],
                    'Ensemble': ensemble_res['Ensemble_30']
                }
                st.plotly_chart(plot_forecast(forecast_30, 30), width='stretch')

# --- Tab 3: Training ---
with tab3:
    st.subheader("⚙️ Dynamic Model Training")
    st.write("Train models dynamically based on the currently loaded stock data.")
    
    model_to_train = st.selectbox("Select Model", ["LSTM", "GRU"])
    
    if st.button(f"Train {model_to_train} Model Now"):
        progress_bar = st.progress(0)
        with st.spinner(f"Training {model_to_train} for {ticker}..."):
            res = load_or_train_dl_model(ticker, model_to_train, df, force_train=True, progress_bar=progress_bar)
            st.success(f"{model_to_train} Model trained successfully and saved to disk!")
            st.json(res['metrics'])

# --- Tab 4: Alerts & History ---
with tab4:
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.subheader("🔔 Set Price Alert")
        alert_price = st.number_input("Target Price", min_value=0.0, value=float(current_price))
        condition = st.selectbox("Condition", ["above", "below"])
        if st.button("Add Alert"):
            add_alert(ticker, alert_price, condition)
            st.success("Alert added!")
            
        st.subheader("Active Alerts")
        st.write(pd.DataFrame(get_active_alerts(), columns=["ID", "Ticker", "Target", "Condition"]))
        
    with col_a2:
        st.subheader("🕒 Recent Search History")
        st.write(pd.DataFrame(get_recent_queries(), columns=["Ticker", "Search Time"]))