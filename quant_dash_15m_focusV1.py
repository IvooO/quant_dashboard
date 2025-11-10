import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import ccxt
import time
from datetime import datetime, timedelta

# --- 1. CONFIGURATION AND INITIALIZATION ---

st.set_page_config(
    page_title="MTF Signal Confluence Monitor", # Updated Name
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- EXCHANGE CONFIGURATION (Using Kraken for reliability) ---
EXCHANGE_ID = 'kraken' 
SYMBOL = 'BTC/USD'     
MAX_TICKS_HISTORY = 20000 
MTF_INTERVAL_15M = '15T'
MTF_INTERVAL_1H = '1H'

# Initialize CCXT and Session State
@st.cache_resource
def initialize_ccxt():
    """Initializes and caches the CCXT exchange object."""
    return getattr(ccxt, EXCHANGE_ID)()

exchange = initialize_ccxt()

# Initialize all necessary session state variables
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = pd.DataFrame()
if 'df_15m_signals' not in st.session_state:
    st.session_state.df_15m_signals = pd.DataFrame()
if 'df_1h_signals' not in st.session_state:
    st.session_state.df_1h_signals = pd.DataFrame()
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = datetime.now() - timedelta(minutes=10)


# --- 2. DATA PROCESSING FUNCTIONS ---

def fetch_raw_trades(symbol, limit):
    """
    Fetches raw trades for historical processing, with fallback for exchanges
    that do not support fetch_trades (using OHLCV as a proxy).
    """
    try:
        # Check if the exchange supports fetch_trades
        if not exchange.has['fetchTrades']:
            st.warning(f"{EXCHANGE_ID} does not support fetching raw trades. Switching to fetch_ohlcv (candles). V-Imb will be inaccurate.")
            
            # Fallback logic: Use 1-minute OHLCV data
            ohlcvs = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=limit)
            if not ohlcvs:
                return st.session_state.trade_history.copy(), 0.0

            df_ohlcv = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_ohlcv['Time'] = pd.to_datetime(df_ohlcv['timestamp'], unit='ms')
            df_ohlcv['Price'] = df_ohlcv['close'] # Use close as the price
            df_ohlcv['Size'] = df_ohlcv['volume'] # Use volume as the size proxy
            df_ohlcv['Side'] = 'BUY' # Side is not available, default to BUY (V-Imb will be ~0)
            
            df_ohlcv = df_ohlcv[['Time', 'Price', 'Size', 'Side']].sort_values('Time').drop_duplicates(subset=['Time', 'Price', 'Size']).reset_index(drop=True)
            df_history = df_ohlcv.tail(MAX_TICKS_HISTORY).reset_index(drop=True)
            st.session_state.trade_history = df_history
            
            current_price = df_history['Price'].iloc[-1] if not df_history.empty else 0.0
            return df_history, current_price
        
        # Original logic (preferred if exchange supports raw trades)
        trades = exchange.fetch_trades(symbol, limit=limit)
        if not trades:
            return st.session_state.trade_history.copy(), 0.0

        df = pd.DataFrame(trades)
        df['Time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['Price'] = df['price']
        df['Size'] = df['amount']
        df['Side'] = df['side'].str.upper()
        
        # Combine new trades with existing, drop duplicates, and trim history
        df = df[['Time', 'Price', 'Size', 'Side']].sort_values('Time').drop_duplicates(subset=['Time', 'Price', 'Size']).reset_index(drop=True)
        st.session_state.trade_history = pd.concat([st.session_state.trade_history, df]).drop_duplicates(subset=['Time', 'Price', 'Size'])
        df_history = st.session_state.trade_history.tail(MAX_TICKS_HISTORY).reset_index(drop=True)
        st.session_state.trade_history = df_history
        
        current_price = df_history['Price'].iloc[-1] if not df_history.empty else 0.0
        return df_history, current_price
    except Exception as e:
        # Catch CCXT errors related to API or symbol issues
        st.error(f"Error fetching data with CCXT: {e}. Double-check the symbol/exchange ID.")
        return st.session_state.trade_history.copy(), 0.0


def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    Calculates MACD, Signal Line, and MACD Signal Status.
    NOTE: DataFrame MUST be in ascending chronological order for correct EMA calculation.
    """
    # 1. Calculate EMAs
    df['EMA_Fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['EMA_Slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    
    # 2. Calculate MACD Line
    df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']
    
    # 3. Calculate Signal Line (EMA of MACD)
    df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    
    # 4. Determine MACD Signal Status (Crossover)
    df['MACD_Cross'] = np.where(df['MACD'] > df['Signal_Line'], 1, -1)
    df['MACD_Signal_Status'] = np.where(
        (df['MACD_Cross'] != df['MACD_Cross'].shift(1)),
        np.where(df['MACD_Cross'] == 1, 'Buy Crossover', 'Sell Crossover'),
        'Hold'
    )
    
    # The current bar's status is simply based on position (MACD > Signal Line)
    # Use .iloc[-1] to safely target the last row by integer position
    last_idx = df.index[-1]
    df.loc[last_idx, 'MACD_Signal_Status'] = np.where(
        df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1], 
        'Buy Momentum', 
        'Sell Momentum'
    )
    
    return df


def get_mtf_signals(df_history, v_imb_threshold, interval):
    """
    Calculates V-Imb and MACD signals over the specified time interval.
    """
    if df_history.empty:
        return pd.DataFrame()

    df_resample = df_history.set_index('Time')

    # --- 1. OHLCV and V-Imb Calculation ---
    ohlcv = df_resample['Price'].resample(interval).ohlc().dropna()
    ohlcv['Volume'] = df_resample['Size'].resample(interval).sum()

    def calculate_imb(group):
        buy_vol = group[group['Side'] == 'BUY']['Size'].sum()
        sell_vol = group[group['Side'] == 'SELL']['Size'].sum()
        total_vol = buy_vol + sell_vol
        imb = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0
        return pd.Series({'V_Imb': imb, 'Total_Aggressive_Vol': total_vol})

    # Only apply V-Imb calculation if Side data exists (i.e., we are using fetch_trades)
    if 'Side' in df_resample.columns and (df_resample['Side'].iloc[-1] in ['BUY', 'SELL']):
        df_mtf_metrics = df_resample.resample(interval).apply(calculate_imb).dropna(subset=['Total_Aggressive_Vol'])
        df_mtf = ohlcv.join(df_mtf_metrics).dropna()
    else:
        # Fallback for OHLCV data where V-Imb is meaningless/set to 0
        df_mtf = ohlcv
        # Fill NaN values in OHLCV generated columns to prevent join failure
        df_mtf = df_mtf.dropna()
        df_mtf['V_Imb'] = 0.0
        df_mtf['Total_Aggressive_Vol'] = df_mtf['Volume']


    # Determine the V-Imb Signal Status
    df_mtf['V_Imb_Signal_Status'] = np.select(
        [
            df_mtf['V_Imb'] > v_imb_threshold,
            df_mtf['V_Imb'] < -v_imb_threshold,
            (df_mtf['V_Imb'].abs() <= v_imb_threshold) & (df_mtf['V_Imb'] > 0),
            (df_mtf['V_Imb'].abs() <= v_imb_threshold) & (df_mtf['V_Imb'] < 0)
        ],
        [
            'Strong Buy',
            'Strong Sell',
            'Weak Buy/Hold',
            'Weak Sell/Hold'
        ],
        default='Neutral/Low Volume'
    )
    
    # --- 2. MACD Calculation ---
    
    # IMPORTANT: MACD must be calculated on ascending data
    df_mtf = df_mtf.sort_index(ascending=True).rename(columns={'close': 'close'})
    
    # Ensure there's enough data for MACD
    if len(df_mtf) > 26: # 26 is the slow period
        df_mtf = calculate_macd(df_mtf)
    else:
        # Not enough data, fill with NaNs or defaults
        df_mtf['MACD'] = 0.0
        df_mtf['Signal_Line'] = 0.0
        df_mtf['MACD_Signal_Status'] = 'N/A'


    # Final preparation: Reset index and sort descending for UI display
    df_mtf = df_mtf.reset_index().rename(columns={'Time': 'Close Time'})
    return df_mtf.sort_values('Close Time', ascending=False)


# --- 3. UI DISPLAY FUNCTIONS ---

def display_signal_table(col, df, interval, max_rows):
    """Displays the V-Imb and MACD signal history table, skipping the current (open) bar."""
    
    col.subheader(f"🟢 {interval} Closed Bar History (Last {max_rows})")

    if df.empty or len(df) < 2:
        col.info(f"Not enough data to display {interval} closed signals.")
        return

    # Skip the first row (the currently forming bar) and take the next `max_rows`
    df_display = df.iloc[1:].head(max_rows).copy()
    
    df_display['V-Imb Value'] = df_display['V_Imb'].apply(lambda x: f"{x:+.3f}")
    df_display['MACD Value'] = df_display['MACD'].apply(lambda x: f"{x:+.2f}")
    df_display['Signal Line'] = df_display['Signal_Line'].apply(lambda x: f"{x:+.2f}")
    df_display['Close Time'] = df_display['Close Time'].dt.strftime('%H:%M:%S')

    def style_v_imb(val):
        if 'Buy' in val: return 'background-color: rgba(0, 128, 0, 0.2); font-weight: bold;'
        if 'Sell' in val: return 'background-color: rgba(255, 0, 0, 0.2); font-weight: bold;'
        return 'background-color: rgba(128, 128, 128, 0.1);'
        
    def style_macd(val):
        if 'Buy Crossover' in val: return 'background-color: rgba(0, 255, 255, 0.2); font-weight: bold;'
        if 'Sell Crossover' in val: return 'background-color: rgba(255, 105, 180, 0.2); font-weight: bold;'
        if 'Hold' in val: return 'background-color: rgba(128, 128, 128, 0.1);'
        return ''
        
    col.dataframe(
        df_display[['Close Time', 'V-Imb Value', 'V_Imb_Signal_Status', 'MACD Value', 'Signal Line', 'MACD_Signal_Status', 'Total_Aggressive_Vol']].style
        .applymap(style_v_imb, subset=['V_Imb_Signal_Status'])
        .applymap(style_macd, subset=['MACD_Signal_Status'])
        .format({'Total_Aggressive_Vol': '{:,.0f}'}),
        use_container_width=True,
        column_config={
            'Total_Aggressive_Vol': st.column_config.ProgressColumn(
                f"{interval} Volume",
                format="%f",
                min_value=0,
                max_value=df_display['Total_Aggressive_Vol'].max() if not df_display.empty else 1,
            ),
            'V-Imb Value': st.column_config.TextColumn("V-Imb"),
            'V_Imb_Signal_Status': st.column_config.TextColumn("V-Imb Status"),
            'MACD Value': st.column_config.TextColumn("MACD"),
            'Signal Line': st.column_config.TextColumn("Signal"),
            'MACD_Signal_Status': st.column_config.TextColumn("MACD Status")
        },
        hide_index=True
    )

# --- 4. STREAMLIT UI EXECUTION ---

st.title(Bitcoin Flow & Momentum Confluence Monitor") # Updated Title
st.markdown(f"### Multi-Factor Signal Analysis: {MTF_INTERVAL_15M} (Execution) & {MTF_INTERVAL_1H} (Confirmation)") # Updated Subtitle

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.subheader("V-Imb Trigger Tuning")
    v_imb_threshold = st.slider("V-Imb Trigger Threshold", 0.05, 0.5, 0.3, 0.01, key='v_imb_threshold')
    st.markdown("---")
    st.write(f"MACD (12/26/9) is automatically calculated.")
    st.markdown("---")
    update_interval = st.slider("UI Update Interval (seconds)", 0.5, 5.0, 1.0)
    st.write(f"Fetching up to **{MAX_TICKS_HISTORY}** trades every **{update_interval}s**.")
    st.subheader("Last Fetch Time")
    st.write(st.session_state.last_fetch_time.strftime("%H:%M:%S"))


# --- Data Fetch and Processing Trigger ---
df_live, current_price = fetch_raw_trades(SYMBOL, limit=MAX_TICKS_HISTORY)

# Generate the MTF signals
df_15m = get_mtf_signals(df_live, v_imb_threshold, interval=MTF_INTERVAL_15M)
df_1h = get_mtf_signals(df_live, v_imb_threshold, interval=MTF_INTERVAL_1H)

# Explicitly store the calculated signals in session state
st.session_state.df_15m_signals = df_15m
st.session_state.df_1h_signals = df_1h

if not df_live.empty and not df_15m.empty and len(df_15m) > 0 and len(df_1h) > 0:
    
    # Get the latest 15M signal metrics (for Key Metrics - this is the OPEN bar)
    latest_15m = df_15m.iloc[0]
    current_imb = latest_15m['V_Imb']
    current_v_imb_signal = latest_15m['V_Imb_Signal_Status']
    current_macd = latest_15m['MACD']
    current_macd_signal = latest_15m['MACD_Signal_Status']
    
    # --- A. KEY TRADING METRICS (15M Focus) ---
    st.subheader(f"Current Bar Metrics ({MTF_INTERVAL_15M} - Open)")
    col1, col2, col3, col4 = st.columns(4)
    
    # KPI 1: Current Price (Last Tick)
    price_delta = df_live['Price'].iloc[-1] - df_live['Price'].iloc[-2] if len(df_live) > 1 else 0.0
    col1.metric("Current Price (Last Tick)", f"${current_price:,.2f}", delta=f"{price_delta:.2f}")
    
    # KPI 2: Current V-Imb (Last 15M Candle)
    imb_delta = current_imb
    imb_color_mode = 'normal' if imb_delta > v_imb_threshold or imb_delta < -v_imb_threshold else 'off'
    col2.metric(f"V-Imb ({MTF_INTERVAL_15M} Open Bar)", f"{current_imb:+.3f}", delta=imb_delta, delta_color=imb_color_mode)
    
    # KPI 3: Trades in History
    col3.metric(f"Trades in History (Max {MAX_TICKS_HISTORY})", f"{len(df_live)}")
    
    # KPI 4: 15M V-Imb Signal Status (Primary Trigger)
    if 'Buy' in current_v_imb_signal:
        signal_emoji = "🟢"; delta_text = f"Strong V-Imb ({MTF_INTERVAL_15M})"
        delta_color_mode = "normal" if 'Strong' in current_v_imb_signal else 'off'
    elif 'Sell' in current_v_imb_signal:
        signal_emoji = "🔴"; delta_text = f"Strong V-Imb ({MTF_INTERVAL_15M})"
        delta_color_mode = "normal"
    else:
        signal_emoji = "🟡"; delta_text = f"Neutral V-Imb ({MTF_INTERVAL_15M})"
        delta_color_mode = "off"
    col4.metric(f"V-Imb Trigger ({MTF_INTERVAL_15M})",
                f"{signal_emoji} {current_v_imb_signal}",
                delta=delta_text,
                delta_color=delta_color_mode
    )
    st.markdown("---")

    # --- B. CHARTS AND VISUALIZATIONS ---
    col_chart_1, col_chart_2 = st.columns(2)

    # 15-Minute Candlestick Chart
    with col_chart_1:
        st.subheader(f"{MTF_INTERVAL_15M} Candlestick Chart with V-Imb Signals")
        fig_candle = go.Figure(data=[go.Candlestick(
            x=df_15m['Close Time'],
            open=df_15m['open'],
            high=df_15m['high'],
            low=df_15m['low'],
            close=df_15m['close'],
            name="Price"
        )])
        buy_signals = df_15m[df_15m['V_Imb_Signal_Status'].str.contains('Strong Buy')]
        sell_signals = df_15m[df_15m['V_Imb_Signal_Status'].str.contains('Strong Sell')]

        if not buy_signals.empty:
            fig_candle.add_trace(go.Scatter(
                x=buy_signals['Close Time'],
                y=buy_signals['low'] * 0.9995,
                mode='markers', name='Buy Signal',
                marker=dict(symbol='triangle-up', size=10, color='green')
            ))

        if not sell_signals.empty:
            fig_candle.add_trace(go.Scatter(
                x=sell_signals['Close Time'],
                y=sell_signals['high'] * 1.0005,
                mode='markers', name='Sell Signal',
                marker=dict(symbol='triangle-down', size=10, color='red')
            ))

        fig_candle.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=450)
        st.plotly_chart(fig_candle, use_container_width=True)

    # Volume Imbalance Meter chart
    with col_chart_2:
        st.subheader(f"Volume Imbalance Meter ({MTF_INTERVAL_15M} Current Bar)")
        imb_df_current = pd.DataFrame({'Metric': ['Volume Imbalance'], 'Value': [current_imb]})
        fig_imb = px.bar(imb_df_current, x='Metric', y='Value',
                         range_y=[-1, 1], height=400,
                         color='Value',
                         color_continuous_scale=px.colors.diverging.RdYlGn,
                         color_continuous_midpoint=0,
                         title=f"V-Imb Thresholds: +/- {v_imb_threshold:.2f}"
        )
        fig_imb.update_layout(template='plotly_dark', showlegend=False)
        fig_imb.add_hline(y=v_imb_threshold, line_dash="dash", line_color="green", annotation_text="Buy Threshold")
        fig_imb.add_hline(y=-v_imb_threshold, line_dash="dash", line_color="red", annotation_text="Sell Threshold")
        st.plotly_chart(fig_imb, use_container_width=True)
    st.markdown("---")

    # --- C. Multi-Timeframe Signal Tables (15M and 1H) ---
    col_15m_table, col_1h_table = st.columns(2)

    # 1. 15M Signal Table (Last 8 CLOSED)
    with col_15m_table:
        display_signal_table(st.container(), st.session_state.df_15m_signals, MTF_INTERVAL_15M, 8)

    # 2. 1H Signal Table (Last 4 CLOSED)
    with col_1h_table:
        display_signal_table(st.container(), st.session_state.df_1h_signals, MTF_INTERVAL_1H, 4)
        
    st.markdown("---")

    # --- D. Multi-Timeframe Signal Summary ---
    
    # D.1 V-Imb Trigger Summary
    st.markdown("### V-Imbalance Trigger Filter")
    col_v_imb_15m, col_v_imb_1h = st.columns(2)
    
    # 15M Open V-Imb Signal
    with col_v_imb_15m:
        if 'Buy' in current_v_imb_signal:
            signal_emoji_15m = "🟢"
            border_color_15m = "green"
        elif 'Sell' in current_v_imb_signal:
            signal_emoji_15m = "🔴"
            border_color_15m = "red"
        else:
            signal_emoji_15m = "🟡"
            border_color_15m = "orange"

        st.subheader(f"{MTF_INTERVAL_15M} Open V-Imb Signal")
        st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; border: 2px solid {border_color_15m}; text-align: center; background-color: #1e1e1e;">
                <p style="font-size: 28px; font-weight: bold; margin-bottom: 5px;">{signal_emoji_15m} {current_v_imb_signal}</p>
                <p style="font-size: 18px; margin-top: 0;">V-Imb: **{current_imb:+.3f}**</p>
                <p style="font-size: 12px; color: #aaa;">Current Bar is forming</p>
            </div>
        """, unsafe_allow_html=True)

    # 1H Closed V-Imb Signal
    with col_v_imb_1h:
        latest_1h_signal = "N/A"
        latest_1h_imb = 0.0
        latest_1h_time = "N/A"
        border_color_1h = "#333"
        signal_emoji_1h = "⚫"

        if len(df_1h) >= 2:
            latest_1h_closed = df_1h.iloc[1]
            latest_1h_signal = latest_1h_closed['V_Imb_Signal_Status']
            latest_1h_imb = latest_1h_closed['V_Imb']
            latest_1h_time = latest_1h_closed['Close Time'].strftime('%H:%M:%S')

            if 'Strong Buy' in latest_1h_signal:
                signal_emoji_1h = "🟢"
                border_color_1h = "green"
            elif 'Strong Sell' in latest_1h_signal:
                signal_emoji_1h = "🔴"
                border_color_1h = "red"
            elif 'Weak' in latest_1h_signal:
                signal_emoji_1h = "🟡"
                border_color_1h = "orange"

        st.subheader(f"{MTF_INTERVAL_1H} Closed V-Imb Signal")
        st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; border: 2px solid {border_color_1h}; text-align: center; background-color: #1e1e1e;">
                <p style="font-size: 28px; font-weight: bold; margin-bottom: 5px;">{signal_emoji_1h} {latest_1h_signal}</p>
                <p style="font-size: 18px; margin-top: 0;">V-Imb: **{latest_1h_imb:+.3f}**</p>
                <p style="font-size: 12px; color: #aaa;">Closed at: {latest_1h_time}</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    
    # D.2 MACD Momentum Filter
    st.markdown("### MACD Momentum Filter (12/26/9)")
    col_macd_15m, col_macd_1h = st.columns(2)
    
    # 15M Open MACD Signal
    with col_macd_15m:
        if 'Buy' in current_macd_signal:
            signal_emoji_15m_macd = "⬆️"
            border_color_15m_macd = "cyan"
        elif 'Sell' in current_macd_signal:
            signal_emoji_15m_macd = "⬇️"
            border_color_15m_macd = "magenta"
        else:
            signal_emoji_15m_macd = "➖"
            border_color_15m_macd = "gray"
            
        st.subheader(f"{MTF_INTERVAL_15M} Open MACD Signal")
        st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; border: 2px solid {border_color_15m_macd}; text-align: center; background-color: #1e1e1e;">
                <p style="font-size: 28px; font-weight: bold; margin-bottom: 5px;">{signal_emoji_15m_macd} {current_macd_signal}</p>
                <p style="font-size: 18px; margin-top: 0;">MACD: **{current_macd:+.2f}**</p>
                <p style="font-size: 12px; color: #aaa;">Current Bar is forming</p>
            </div>
        """, unsafe_allow_html=True)

    # 1H Closed MACD Signal
    with col_macd_1h:
        latest_1h_macd_signal = "N/A"
        latest_1h_macd = 0.0
        latest_1h_time = "N/A"
        border_color_1h_macd = "#333"
        signal_emoji_1h_macd = "⚫"

        if len(df_1h) >= 2:
            latest_1h_closed = df_1h.iloc[1]
            latest_1h_macd_signal = latest_1h_closed['MACD_Signal_Status']
            latest_1h_macd = latest_1h_closed['MACD']
            latest_1h_time = latest_1h_closed['Close Time'].strftime('%H:%M:%S')

            if 'Buy Crossover' in latest_1h_macd_signal:
                signal_emoji_1h_macd = "⬆️"
                border_color_1h_macd = "cyan"
            elif 'Sell Crossover' in latest_1h_macd_signal:
                signal_emoji_1h_macd = "⬇️"
                border_color_1h_macd = "magenta"
            elif 'Hold' in latest_1h_macd_signal:
                signal_emoji_1h_macd = "➖"
                border_color_1h_macd = "gray"
            else: # Momentum status for the closed bar, not a fresh cross
                 if latest_1h_macd > latest_1h_closed['Signal_Line']:
                    signal_emoji_1h_macd = "⬆️"
                    border_color_1h_macd = "cyan"
                    latest_1h_macd_signal = 'Buy Momentum (Hold)'
                 else:
                    signal_emoji_1h_macd = "⬇️"
                    border_color_1h_macd = "magenta"
                    latest_1h_macd_signal = 'Sell Momentum (Hold)'

        st.subheader(f"{MTF_INTERVAL_1H} Closed MACD Signal")
        st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; border: 2px solid {border_color_1h_macd}; text-align: center; background-color: #1e1e1e;">
                <p style="font-size: 28px; font-weight: bold; margin-bottom: 5px;">{signal_emoji_1h_macd} {latest_1h_macd_signal}</p>
                <p style="font-size: 18px; margin-top: 0;">MACD: **{latest_1h_macd:+.2f}**</p>
                <p style="font-size: 12px; color: #aaa;">Closed at: {latest_1h_time}</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("---")


else:
    st.info(f"Attempting to connect to {EXCHANGE_ID.upper()} and fetch initial trade data for {SYMBOL}...")


# --- 5. Rerun Logic ---

time.sleep(update_interval)
st.session_state.last_fetch_time = datetime.now()
st.rerun()
