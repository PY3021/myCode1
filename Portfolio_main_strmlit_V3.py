import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import yfinance as yf
import os
import pickle
from datetime import datetime, date

# --- 1. Page Configuration ---
st.set_page_config(page_title="Consolidated Holdings", layout="wide", initial_sidebar_state="expanded")

CACHE_FILE = "portfolio_cache.pkl"

# --- 2. Live Market Data Fetcher ---
def fetch_market_data(symbols):
    market_data = {}
    for sym in symbols:
        ticker_sym = sym if sym.endswith('.NS') else f"{sym}.NS" 
        try:
            ticker = yf.Ticker(ticker_sym)
            hist = ticker.history(period="max")
            hist_2d = hist.tail(2) if not hist.empty else pd.DataFrame()
            
            if len(hist_2d) >= 2:
                prev_close = float(hist_2d['Close'].iloc[0])
                ltp = float(hist_2d['Close'].iloc[1])
            elif len(hist_2d) == 1:
                prev_close, ltp = float(hist_2d['Close'].iloc[0]), float(hist_2d['Close'].iloc[0])
            else:
                prev_close, ltp = 0.0, 0.0
                
            info = ticker.info
            market_data[sym] = {
                'LTP': ltp, 
                'Prev_Close': prev_close,
                'Market_Cap': info.get('marketCap', 0),
                '52W_High': info.get('fiftyTwoWeekHigh', 0),
                '52W_Low': info.get('fiftyTwoWeekLow', 0),
                'ATH': hist['High'].max() if not hist.empty else 0
            }
        except Exception:
            market_data[sym] = {'LTP': 0.0, 'Prev_Close': 0.0, 'Market_Cap': 0, '52W_High': 0, '52W_Low': 0, 'ATH': 0}
    return market_data

# --- Custom XIRR Calculator ---
def calculate_xirr(cash_flows):
    """Calculates Extended Internal Rate of Return (XIRR)"""
    if len(cash_flows) < 2: return 0.0
    
    cash_flows = sorted(cash_flows, key=lambda x: x['date'])
    
    # Must have both positive and negative cash flows to calculate XIRR
    if not any(cf['amount'] > 0 for cf in cash_flows) or not any(cf['amount'] < 0 for cf in cash_flows):
        return 0.0
        
    d0 = cash_flows[0]['date']
    
    def xnpv(rate):
        if rate <= -1.0: return float('inf')
        return sum(cf['amount'] / ((1.0 + rate) ** ((cf['date'] - d0).days / 365.0)) for cf in cash_flows)
        
    left, right = -0.9999, 10000.0 # Bisection bounds bounds (-99.99% to 1,000,000%)
    for _ in range(100):
        mid = (left + right) / 2.0
        if xnpv(mid) > 0:
            left = mid
        else:
            right = mid
    return mid * 100.0

# --- 3. FIFO Logic Engine ---
def calculate_fifo(group):
    buys = []  
    realized_pnl = 0.0
    group = group.sort_values(by='Date')
    
    for _, row in group.iterrows():
        qty = float(row['Quantity'])
        rate = float(row['Rate of purchase'])
        action = str(row['Type']).strip().title()
        
        if action == 'Buy':
            buys.append({'qty': qty, 'rate': rate})
        elif action == 'Sell':
            sell_qty = qty
            while sell_qty > 0 and buys:
                oldest_buy = buys[0]
                if oldest_buy['qty'] <= sell_qty:
                    matched_qty = oldest_buy['qty']
                    realized_pnl += matched_qty * (rate - oldest_buy['rate'])
                    sell_qty -= matched_qty
                    buys.pop(0)  
                else:
                    matched_qty = sell_qty
                    realized_pnl += matched_qty * (rate - oldest_buy['rate'])
                    oldest_buy['qty'] -= matched_qty
                    sell_qty = 0
                    
    holding_qty = sum(b['qty'] for b in buys)
    holding_invested = sum(b['qty'] * b['rate'] for b in buys)
    avg_price = holding_invested / holding_qty if holding_qty > 0 else 0.0
    
    return pd.Series({
        'Total_Qty': holding_qty,
        'Avg_Price': avg_price,
        'Total_Invested': holding_invested,
        'Realized_PnL': realized_pnl
    })

# --- 4. Data Generation & Saving ---
def generate_and_save_data():
    raw_df = pd.read_excel('trade.xlsx')
    
    holdings = raw_df.groupby(['Account', 'Symbol']).apply(calculate_fifo).reset_index()
    holdings = holdings[holdings['Total_Qty'] > 0].copy()
    
    unique_symbols = holdings['Symbol'].unique().tolist()
    live_prices = fetch_market_data(unique_symbols)
    
    holdings['LTP'] = holdings['Symbol'].map(lambda x: live_prices[x]['LTP'])
    holdings['Prev_Close'] = holdings['Symbol'].map(lambda x: live_prices[x]['Prev_Close'])
    
    holdings['Current_Value'] = holdings['Total_Qty'] * holdings['LTP']
    holdings['Unrealized_PnL'] = holdings['Current_Value'] - holdings['Total_Invested']
    holdings['Day_PnL'] = (holdings['LTP'] - holdings['Prev_Close']) * holdings['Total_Qty']
    
    holdings['% Unrealised_PnL'] = np.where(holdings['Total_Invested'] > 0, (holdings['Unrealized_PnL'] / holdings['Total_Invested']) * 100, 0.0)
    holdings['% Daily_PnL'] = np.where(holdings['Prev_Close'] > 0, ((holdings['LTP'] - holdings['Prev_Close']) / holdings['Prev_Close']) * 100, 0.0)
    
    update_time = datetime.now().strftime("%d %b %Y, %I:%M %p")
    
    cache_data = {
        "update_time": update_time,
        "holdings": holdings,
        "raw_data": raw_df,
        "market_data": live_prices
    }
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache_data, f)
        
    return holdings, raw_df, live_prices, update_time

# --- 5. Application Flow & Data Loading ---
st.sidebar.header("Data Sync")

if st.sidebar.button("🔄 Refresh Live Data", use_container_width=True):
    with st.spinner("Fetching live prices and recalculating..."):
        holdings_df, raw_df, live_prices, last_updated = generate_and_save_data()
        st.sidebar.success("Data refreshed successfully!")
else:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cache_data = pickle.load(f)
            holdings_df = cache_data["holdings"]
            raw_df = cache_data["raw_data"]
            live_prices = cache_data["market_data"]
            last_updated = cache_data["update_time"]
    else:
        with st.spinner("Initializing dashboard..."):
            holdings_df, raw_df, live_prices, last_updated = generate_and_save_data()

st.sidebar.caption(f"**Last Updated:** {last_updated}")
st.sidebar.markdown("---")

# ==========================================
# PAGE 2: SYMBOL DEEP DIVE (DETAIL VIEW)
# ==========================================
if "symbol" in st.query_params:
    selected_sym = st.query_params["symbol"]
    
    if st.button("⬅️ Back to Main Dashboard"):
        st.query_params.clear()
        st.rerun()
        
    st.title(f"Stock Deep Dive: {selected_sym}")
    st.markdown("---")
    
    # 1. Market Data Fetching
    sym_market_data = live_prices.get(selected_sym, {})
    mkt_cap = sym_market_data.get('Market_Cap', 0)
    mkt_cap_cr = mkt_cap / 10000000  # Convert to Crores
    
    ath = sym_market_data.get('ATH', 0)
    high_52 = sym_market_data.get('52W_High', 0)
    low_52 = sym_market_data.get('52W_Low', 0)
    
    # 2. Transaction Aggregations
    filtered_raw = raw_df[raw_df['Symbol'] == selected_sym].copy()
    buy_df = filtered_raw[filtered_raw['Type'].str.title() == 'Buy']
    sell_df = filtered_raw[filtered_raw['Type'].str.title() == 'Sell']
    
    total_qty_purchased = buy_df['Quantity'].sum()
    total_amount_purchased = (buy_df['Quantity'] * buy_df['Rate of purchase']).sum()
    total_qty_sold = sell_df['Quantity'].sum()
    
    # 3. Holding Aggregations
    sym_holdings = holdings_df[holdings_df['Symbol'] == selected_sym]
    holding_qty = sym_holdings['Total_Qty'].sum()
    holding_value = sym_holdings['Current_Value'].sum()
    realized_pnl = sym_holdings['Realized_PnL'].sum()
    unrealized_pnl = sym_holdings['Unrealized_PnL'].sum()
    total_pnl = realized_pnl + unrealized_pnl
    pct_total_pnl = (total_pnl / total_amount_purchased * 100) if total_amount_purchased > 0 else 0
    
    # 4. Calculate XIRR for Active Holding
    remaining_buys = []
    cash_flows = []
    sym_raw_sorted = filtered_raw.sort_values(by='Date')
    
    for _, row in sym_raw_sorted.iterrows():
        qty = float(row['Quantity'])
        rate = float(row['Rate of purchase'])
        action = str(row['Type']).strip().title()
        txn_date = pd.to_datetime(row['Date']).date()
        
        if action == 'Buy':
            remaining_buys.append({'date': txn_date, 'qty': qty, 'rate': rate})
        elif action == 'Sell':
            sell_qty = qty
            while sell_qty > 0 and remaining_buys:
                oldest = remaining_buys[0]
                if oldest['qty'] <= sell_qty:
                    sell_qty -= oldest['qty']
                    remaining_buys.pop(0)
                else:
                    oldest['qty'] -= sell_qty
                    sell_qty = 0
                    
    for b in remaining_buys:
        cash_flows.append({'date': b['date'], 'amount': -1 * b['qty'] * b['rate']})
        
    if holding_qty > 0:
        cash_flows.append({'date': datetime.now().date(), 'amount': holding_qty * sym_market_data.get('LTP', 0)})
        xirr_pct = calculate_xirr(cash_flows)
    else:
        xirr_pct = 0.0

    xirr_color = "#16a34a" if xirr_pct >= 0 else "#dc2626"
    
    # 5. Detail Tiles HTML (Zero Indentation)
    html_details = f"""
<div style="display: flex; gap: 20px; margin-bottom: 20px;">
<div style="flex: 1; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; background-color: #f8fafc;">
<p style="margin: 0; color: #64748b; font-size: 16px;">Market Capital</p>
<h3 style="margin: 8px 0 0 0; color: #0f172a; font-size: 28px;">₹{mkt_cap_cr:,.2f} Cr</h3>
</div>
<div style="flex: 1; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; background-color: #f8fafc;">
<p style="margin: 0; color: #64748b; font-size: 16px;">All Time High (ATH)</p>
<h3 style="margin: 8px 0 0 0; color: #0f172a; font-size: 28px;">₹{ath:,.2f}</h3>
</div>
<div style="flex: 1; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; background-color: #f8fafc;">
<p style="margin: 0; color: #64748b; font-size: 16px;">52 Week High / Low</p>
<h3 style="margin: 8px 0 0 0; color: #0f172a; font-size: 28px;">₹{high_52:,.2f} <span style="font-size:18px; color:#64748b;">/ ₹{low_52:,.2f}</span></h3>
</div>
</div>
<div style="display: flex; gap: 20px; margin-bottom: 20px;">
<div style="flex: 1; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; background-color: #eff6ff;">
<p style="margin: 0; color: #64748b; font-size: 16px;">Total Qty Purchased & Amount</p>
<h3 style="margin: 8px 0 0 0; color: #0f172a; font-size: 24px;">{int(total_qty_purchased):,} <span style="font-size:16px; color:#64748b;">shares</span> | ₹{total_amount_purchased:,.2f}</h3>
</div>
<div style="flex: 1; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; background-color: #fffbeb;">
<p style="margin: 0; color: #64748b; font-size: 16px;">Total Qty Sold</p>
<h3 style="margin: 8px 0 0 0; color: #0f172a; font-size: 24px;">{int(total_qty_sold):,} <span style="font-size:16px; color:#64748b;">shares</span></h3>
</div>
<div style="flex: 1; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; background-color: #eff6ff;">
<p style="margin: 0; color: #64748b; font-size: 16px;">Holding Qty & Value</p>
<h3 style="margin: 8px 0 0 0; color: #0f172a; font-size: 24px;">{int(holding_qty):,} <span style="font-size:16px; color:#64748b;">shares</span> | ₹{holding_value:,.2f}</h3>
</div>
</div>
<div style="display: flex; gap: 20px; margin-bottom: 30px;">
<div style="flex: 1; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; background-color: #f0fdf4;">
<p style="margin: 0; color: #64748b; font-size: 16px;">Realised & Unrealised PnL</p>
<h3 style="margin: 8px 0 0 0; color: #0f172a; font-size: 24px;">₹{realized_pnl:,.2f} <span style="font-size:16px; color:#64748b;">/</span> ₹{unrealized_pnl:,.2f}</h3>
</div>
<div style="flex: 1; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; background-color: #f0fdf4;">
<p style="margin: 0; color: #64748b; font-size: 16px;">Total PnL & % Return</p>
<h3 style="margin: 8px 0 0 0; color: #0f172a; font-size: 24px;">₹{total_pnl:,.2f} <span style="font-size:16px; color:#64748b;">({pct_total_pnl:,.2f}%)</span></h3>
</div>
<div style="flex: 1; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; background-color: #f8fafc;">
<p style="margin: 0; color: #64748b; font-size: 16px;">Holding XIRR</p>
<h3 style="margin: 8px 0 0 0; color: {xirr_color}; font-size: 24px; font-weight: bold;">{xirr_pct:,.2f}%</h3>
</div>
</div>
"""
    st.markdown(html_details, unsafe_allow_html=True)
    
    # 6. Transaction History Grid
    st.subheader("Transaction History")
    filtered_raw['Date'] = pd.to_datetime(filtered_raw['Date']).dt.strftime('%Y-%m-%d')
    st.dataframe(filtered_raw.sort_values(by='Date', ascending=False), use_container_width=True, hide_index=True)


# ==========================================
# PAGE 1: MAIN DASHBOARD
# ==========================================
else:
    st.sidebar.header("Dashboard Filters")
    account_list = ["All Accounts"] + list(holdings_df['Account'].unique())
    selected_account = st.sidebar.selectbox("Select Account", account_list)

    if selected_account != "All Accounts":
        display_df = holdings_df[holdings_df['Account'] == selected_account]
    else:
        display_df = holdings_df.copy()

    total_invested = display_df['Total_Invested'].sum()
    current_value = display_df['Current_Value'].sum()
    total_unrealized = display_df['Unrealized_PnL'].sum()
    total_realized = display_df['Realized_PnL'].sum()
    total_day_pnl = display_df['Day_PnL'].sum()
    total_overall_pnl = total_unrealized + total_realized

    st.title(f"Portfolio Dashboard: {selected_account}")

    unrealized_pct = (total_unrealized / total_invested * 100) if total_invested > 0 else 0
    unrealized_color = "#16a34a" if total_unrealized >= 0 else "#dc2626"
    unrealized_arrow = "↑" if total_unrealized >= 0 else "↓"

    current_pct = ((current_value - total_invested) / total_invested * 100) if total_invested > 0 else 0
    current_color = "#16a34a" if current_value >= total_invested else "#dc2626"
    current_bg = "#ffffff"  
    current_arrow = "↑" if current_value >= total_invested else "↓"

    day_color = "#16a34a" if total_day_pnl >= 0 else "#dc2626"
    day_arrow = "↑" if total_day_pnl >= 0 else "↓"

    realized_color = "#16a34a" if total_realized >= 0 else "#dc2626"
    realized_arrow = "↑" if total_realized >= 0 else "↓"
    realized_text = "Booked Profit" if total_realized >= 0 else "Booked Loss"

    overall_pct = (total_overall_pnl / total_invested * 100) if total_invested > 0 else 0
    overall_color = "#16a34a" if total_overall_pnl >= 0 else "#dc2626"
    overall_arrow = "↑" if total_overall_pnl >= 0 else "↓"

    tile_bg_invested = "#b2b2ff" 
    tile_bg_current = "#b2b2ff" 
    color_profit = "#30d16d"
    color_loss = "#f74a4a"

    tile_bg_day = color_profit if total_day_pnl >= 0 else color_loss
    tile_bg_unrealized = color_profit if total_unrealized >= 0 else color_loss
    tile_bg_realized = color_profit if total_realized >= 0 else color_loss
    tile_bg_overall = color_profit if total_overall_pnl >= 0 else color_loss
    pill_bg_colored_tiles = "#ffffff"

    # HTML structure (Zero Indentation)
    html_dashboard = f"""
<div style="background-color: #f0f4f8; padding: 30px; border-radius: 15px; box-shadow: inset 0 2px 4px 0 rgba(0,0,0,0.05); border: 1px solid #e2e8f0; margin-bottom: 30px;">
<div style="display: flex; gap: 20px; margin-bottom: 20px;">
<div style="flex: 1; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); background-color: {tile_bg_invested};">
<p style="margin: 0; color: #334155; font-size: 20px; font-weight: 500; font-family: sans-serif;">Total Investment</p>
<h3 style="margin: 8px 0 0 0; color: #0f172a; font-size: 38px; font-weight: 700; font-family: sans-serif;">₹{total_invested:,.2f}</h3>
</div>
<div style="flex: 1; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); background-color: {tile_bg_current};">
<p style="margin: 0; color: #334155; font-size: 20px; font-weight: 500; font-family: sans-serif;">Current Value</p>
<h3 style="margin: 8px 0 12px 0; color: #0f172a; font-size: 38px; font-weight: 700; font-family: sans-serif;">₹{current_value:,.2f}</h3>
<span style="background-color: {current_bg}; color: {current_color}; padding: 4px 10px; border-radius: 20px; font-size: 18px; font-weight: 600; font-family: sans-serif;">{current_arrow} {abs(current_pct):.2f}%</span>
</div>
<div style="flex: 1; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); background-color: {tile_bg_day};">
<p style="margin: 0; color: #f1f5f9; font-size: 20px; font-weight: 500; font-family: sans-serif;">Day's PnL</p>
<h3 style="margin: 8px 0 12px 0; color: #ffffff; font-size: 38px; font-weight: 700; font-family: sans-serif;">₹{total_day_pnl:,.2f}</h3>
<span style="background-color: {pill_bg_colored_tiles}; color: {day_color}; padding: 4px 10px; border-radius: 20px; font-size: 18px; font-weight: 600; font-family: sans-serif;">{day_arrow} ₹{abs(total_day_pnl):,.2f}</span>
</div>
</div>
<div style="display: flex; gap: 20px;">
<div style="flex: 1; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); background-color: {tile_bg_unrealized};">
<p style="margin: 0; color: #f1f5f9; font-size: 20px; font-weight: 500; font-family: sans-serif;">Unrealized PnL</p>
<h3 style="margin: 8px 0 12px 0; color: #ffffff; font-size: 38px; font-weight: 700; font-family: sans-serif;">₹{total_unrealized:,.2f}</h3>
<span style="background-color: {pill_bg_colored_tiles}; color: {unrealized_color}; padding: 4px 10px; border-radius: 20px; font-size: 18px; font-weight: 600; font-family: sans-serif;">{unrealized_arrow} {abs(unrealized_pct):.2f}%</span>
</div>
<div style="flex: 1; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); background-color: {tile_bg_realized};">
<p style="margin: 0; color: #f1f5f9; font-size: 20px; font-weight: 500; font-family: sans-serif;">Realized PnL (Booked)</p>
<h3 style="margin: 8px 0 12px 0; color: #ffffff; font-size: 38px; font-weight: 700; font-family: sans-serif;">₹{total_realized:,.2f}</h3>
<span style="background-color: {pill_bg_colored_tiles}; color: {realized_color}; padding: 4px 10px; border-radius: 20px; font-size: 18px; font-weight: 600; font-family: sans-serif;">{realized_arrow} {realized_text}</span>
</div>
<div style="flex: 1; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); background-color: {tile_bg_overall};">
<p style="margin: 0; color: #f1f5f9; font-size: 20px; font-weight: 500; font-family: sans-serif;">Overall PnL (Realized + Unrealized)</p>
<h3 style="margin: 8px 0 12px 0; color: #ffffff; font-size: 38px; font-weight: 700; font-family: sans-serif;">₹{total_overall_pnl:,.2f}</h3>
<span style="background-color: {pill_bg_colored_tiles}; color: {overall_color}; padding: 4px 10px; border-radius: 20px; font-size: 18px; font-weight: 600; font-family: sans-serif;">{overall_arrow} {abs(overall_pct):.2f}%</span>
</div>
</div>
</div>
"""
    st.markdown(html_dashboard, unsafe_allow_html=True)

    st.subheader("Active Holdings")

    display_table = display_df[['Symbol', 'Total_Qty', 'Avg_Price', 'LTP', 
                                'Total_Invested', 'Current_Value', 'Day_PnL', '% Daily_PnL', 'Unrealized_PnL', '% Unrealised_PnL']].copy()
    
    display_table['Total_Qty'] = display_table['Total_Qty'].astype(int)
    display_table['Symbol'] = "/?symbol=" + display_table['Symbol']

    st.dataframe(
        display_table.style.format({
            'Total_Qty': '{:,}', 
            'Avg_Price': '₹{:.2f}',
            'LTP': '₹{:.2f}',
            'Total_Invested': '₹{:,.2f}',
            'Current_Value': '₹{:,.2f}',
            'Day_PnL': '₹{:,.2f}',
            '% Daily_PnL': '{:.2f}%',
            'Unrealized_PnL': '₹{:,.2f}',
            '% Unrealised_PnL': '{:.2f}%'
        }).map(lambda x: 'color: #30d16d; font-weight: bold;' if x > 0 else 'color: #f74a4a; font-weight: bold;' if x < 0 else '', 
               subset=['Day_PnL', '% Daily_PnL', 'Unrealized_PnL', '% Unrealised_PnL']),
        column_config={
            "Symbol": st.column_config.LinkColumn(
                "Symbol (Click for Details)", 
                display_text=r"^\/\?symbol=(.*)$"
            )
        },
        use_container_width=True,
        hide_index=True
    )