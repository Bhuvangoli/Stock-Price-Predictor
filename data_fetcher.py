import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import feedparser
from textblob import TextBlob
import streamlit as st
import datetime

@st.cache_data(ttl=300) # Cache for 5 minutes
def fetch_stock_data(ticker, period='1y'):
    """Fetch historical stock data."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def add_technical_indicators(df):
    """Add RSI, MACD, and Bollinger Bands to the DataFrame."""
    if df is None or len(df) < 30:
        return df
    
    # RSI
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()
    
    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    
    return df

@st.cache_data(ttl=1800) # Cache news for 30 mins
def fetch_news_sentiment(ticker):
    """Fetch news for a ticker and calculate average sentiment."""
    url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US'
    feed = feedparser.parse(url)
    
    news_items = []
    total_sentiment = 0
    count = 0
    
    for entry in feed.entries[:10]: # Get top 10 news
        title = entry.title
        link = entry.link
        published = entry.published
        
        # Sentiment Analysis
        blob = TextBlob(title)
        sentiment_score = blob.sentiment.polarity
        
        # Classify sentiment
        if sentiment_score > 0.1:
            sentiment_label = "Positive"
        elif sentiment_score < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
            
        total_sentiment += sentiment_score
        count += 1
        
        news_items.append({
            'title': title,
            'link': link,
            'published': published,
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label
        })
    
    avg_sentiment = total_sentiment / count if count > 0 else 0
    overall_sentiment = "Neutral"
    if avg_sentiment > 0.1:
        overall_sentiment = "Positive"
    elif avg_sentiment < -0.1:
        overall_sentiment = "Negative"
        
    return {
        'items': news_items,
        'avg_score': avg_sentiment,
        'overall': overall_sentiment
    }

def get_recommendation(df, avg_sentiment):
    """Generate Buy/Sell/Hold recommendation based on indicators and sentiment."""
    if df is None or len(df) == 0:
        return "Hold", "Not enough data"
        
    latest = df.iloc[-1]
    signals = []
    buy_score = 0
    sell_score = 0
    
    # RSI
    if not pd.isna(latest.get('RSI')):
        if latest['RSI'] < 30:
            signals.append("RSI indicates Oversold")
            buy_score += 1
        elif latest['RSI'] > 70:
            signals.append("RSI indicates Overbought")
            sell_score += 1
            
    # MACD
    if not pd.isna(latest.get('MACD_Diff')):
        if latest['MACD_Diff'] > 0:
            signals.append("MACD is Bullish")
            buy_score += 1
        else:
            signals.append("MACD is Bearish")
            sell_score += 1
            
    # Bollinger Bands
    if not pd.isna(latest.get('BB_Low')) and not pd.isna(latest.get('BB_High')):
        if latest['Close'] < latest['BB_Low']:
            signals.append("Price below lower Bollinger Band")
            buy_score += 1
        elif latest['Close'] > latest['BB_High']:
            signals.append("Price above upper Bollinger Band")
            sell_score += 1
            
    # Sentiment
    if avg_sentiment > 0.1:
        signals.append("News sentiment is Positive")
        buy_score += 1
    elif avg_sentiment < -0.1:
        signals.append("News sentiment is Negative")
        sell_score += 1
        
    if buy_score > sell_score + 1:
        recommendation = "BUY"
    elif sell_score > buy_score + 1:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"
        
    return recommendation, signals
