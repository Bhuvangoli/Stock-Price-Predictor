import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

def plot_candlestick(df, ticker):
    """Plot interactive candlestick chart with Bollinger Bands and Volume."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{ticker} Price', 'Volume'), 
                        row_width=[0.2, 0.7])

    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Price'), 
                  row=1, col=1)

    # Bollinger Bands
    if 'BB_High' in df.columns and 'BB_Low' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], 
                                 line=dict(color='rgba(250, 250, 250, 0.3)', width=1), 
                                 name='BB High'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], 
                                 line=dict(color='rgba(250, 250, 250, 0.3)', width=1), 
                                 name='BB Low', fill='tonexty', fillcolor='rgba(250, 250, 250, 0.05)'), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], showlegend=False, marker_color='rgba(0, 150, 250, 0.8)'), row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

def plot_indicators(df):
    """Plot RSI and MACD."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, subplot_titles=('RSI', 'MACD'))
                        
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Diff'], name='Histogram'), row=2, col=1)
    
    fig.update_layout(template='plotly_dark', height=400, margin=dict(l=0, r=0, t=30, b=0))
    return fig

def plot_predictions(y_true, y_pred_dict, title="Model Predictions vs Actual"):
    """Plot actual vs predicted for multiple models."""
    fig = go.Figure()
    
    # Generate mock dates for validation set
    x_val = list(range(len(y_true)))
    
    fig.add_trace(go.Scatter(x=x_val, y=y_true.flatten(), mode='lines', name='Actual', line=dict(color='white', width=2)))
    
    colors = ['cyan', 'magenta', 'yellow', 'orange']
    for i, (name, y_pred) in enumerate(y_pred_dict.items()):
        fig.add_trace(go.Scatter(x=x_val, y=y_pred.flatten(), mode='lines', name=name, line=dict(color=colors[i % len(colors)], dash='dash')))
        
    fig.update_layout(template='plotly_dark', title=title, height=400, margin=dict(l=0, r=0, t=40, b=0))
    return fig

def plot_forecast(forecast_dict, days=7):
    """Plot future forecast."""
    fig = go.Figure()
    x_future = list(range(1, days + 1))
    
    colors = ['cyan', 'magenta', 'yellow', 'orange']
    for i, (name, y_pred) in enumerate(forecast_dict.items()):
        fig.add_trace(go.Scatter(x=x_future, y=y_pred, mode='lines+markers', name=name, line=dict(color=colors[i % len(colors)])))
        
    fig.update_layout(template='plotly_dark', title=f"Future {days} Days Forecast", height=400, xaxis_title="Days Ahead", yaxis_title="Price")
    return fig

def metric_card(title, value, delta=None):
    """Custom metric card styling."""
    st.metric(label=title, value=value, delta=delta)
    
def display_signal(signal):
    """Display buy/sell/hold signal with color."""
    color = "green" if signal == "BUY" else "red" if signal == "SELL" else "gray"
    st.markdown(f"<h3 style='text-align: center; color: {color}; border: 2px solid {color}; padding: 10px; border-radius: 5px;'>{signal}</h3>", unsafe_allow_html=True)
