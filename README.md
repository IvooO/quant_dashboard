Real-Time Crypto Trading Dashboards

This repository contains a collection of professional-grade trading dashboards built with Streamlit, Plotly, and CCXT. These tools are designed for real-time, multi-timeframe analysis of cryptocurrency markets, focusing on quantitative signals like Volume Imbalance, Momentum (MACD), and Divergence.

Dashboards Included

1. MTF Signal Confluence Monitor (mtf_confluence_monitor.py)

This dashboard provides a high-level, multi-timeframe (MTF) view for signal confirmation, ideal for aligning execution with a higher-level trend.

Primary Focus: 15-Minute (Execution) vs. 1-Hour (Confirmation).

Key Signals:

Volume Imbalance (V-Imb): Calculates aggressive buy/sell pressure for each 15M and 1H candle.

MACD (12, 26, 9): Tracks momentum and crossover signals for both timeframes.

Features:

Displays the signal status for the current open 15M bar (for real-time execution).

Displays the signal status for the last closed 1H bar (for trend confirmation).

Provides history tables for the last 8 closed 15M signals and 4 closed 1H signals.

Dynamically fetches data from Kraken (BTC/USD).

2. MACD Divergence Dashboard (macd_divergence_dashboard.py)

This dashboard is a classic technical analysis tool designed to automatically spot trend reversals and momentum shifts by comparing price action with the MACD oscillator.

Key Signals:

Regular Divergence: Automatically plots ⭐️ Bullish Divergence (Price Lower Low, MACD Higher Low) and ⭐️ Bearish Divergence (Price Higher High, MACD Lower High).

MACD Crossovers (Torys Signals): Plots 🔼 Bullish Crosses (MACD over Signal) and 🔽 Bearish Crosses (MACD under Signal).

Zero Line Crossovers: Plots 🟢 Bullish Trend (MACD over 0) and 🔴 Bearish Trend (MACD under 0) signals.

Features:

Uses yfinance to fetch daily data for any ticker (e.g., BTC-USD, AAPL).

Interactive Plotly chart with all signals plotted.

Adjustable pivot lookback period to fine-tune divergence sensitivity.

Includes a "MACD Trading Playbook" expander to explain how to read the signals.

How to Run

1. Prerequisites

Python 3.10 or newer

pip and venv

2. Setup (Recommended)

This project requires specific package versions. Using a virtual environment is the best way to avoid dependency errors (like the externally-managed-environment error).

1. Clone the repository:

git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
cd YOUR_REPOSITORY


2. Create a virtual environment:
(Use your specific Python command, e.g., python3.12 or python3)

python3 -m venv venv


3. Activate the virtual environment:

On macOS/Linux:

source venv/bin/activate


On Windows:

.\venv\Scripts\activate


4. Install dependencies:
With your virtual environment active, install all required packages from the requirements.txt file.

pip install -r requirements.txt


3. Run the Dashboards

To run the MTF Confluence Monitor:

streamlit run mtf_confluence_monitor.py


To run the MACD Divergence Dashboard:

streamlit run macd_divergence_dashboard.py

