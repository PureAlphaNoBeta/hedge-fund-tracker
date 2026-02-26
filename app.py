import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="13F Historical Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("Hedge Fund 13F Tracker: 10-Year History")

# --- FUND CLASSIFICATION METADATA ---
FUND_METADATA = {
    "XN LP": {"Sector": "Generalist", "Strategy": "Long-Biased"},
    "Voyager Global Management": {"Sector": "Technology", "Strategy": "Long-Biased"},
    "Foxhaven Asset Management": {"Sector": "Technology", "Strategy": "L/S Equity"},
    "Lone Pine": {"Sector": "Generalist", "Strategy": "L/S Equity"},
    "D1 Capital": {"Sector": "Generalist", "Strategy": "L/S Equity"},
    "Soroban Capital": {"Sector": "Generalist", "Strategy": "Long-Biased"},
    "Avala Global": {"Sector": "Technology", "Strategy": "L/S Equity"},
    "Viking Global": {"Sector": "Generalist", "Strategy": "L/S Equity"},
    "SurgoCap": {"Sector": "Generalist", "Strategy": "L/S Equity"},
    "Coatue Management": {"Sector": "Technology", "Strategy": "L/S Equity"},
    "Estuary Capital Management": {"Sector": "Generalist", "Strategy": "L/S Equity"},
    "Slate Path Capital": {"Sector": "Generalist", "Strategy": "L/S Equity"},
    "Maverick Capital": {"Sector": "Generalist", "Strategy": "L/S Equity"},
    "Long Pond Capital": {"Sector": "Real Estate", "Strategy": "L/S Equity"},
    "Sachem Head Capital": {"Sector": "Generalist", "Strategy": "Long-Biased"},
    "RTW Investments": {"Sector": "Biotech", "Strategy": "Long-Biased"},
    "Darwin Global Management": {"Sector": "Biotech", "Strategy": "Long-Biased"},
    "Exome Asset Management": {"Sector": "Biotech", "Strategy": "Long-Biased"},
    "Commodore Capital": {"Sector": "Biotech", "Strategy": "Long-Biased"},
    "Vestal Point Capital": {"Sector": "Biotech", "Strategy": "L/S Equity"},
    "Baker Bros. Advisors": {"Sector": "Biotech", "Strategy": "Long-Biased"},
    "Conversant Capital": {"Sector": "Real Estate", "Strategy": "Long-Biased"},
    "Pentwater Capital Management": {"Sector": "Generalist", "Strategy": "Event-Driven"},
    "Decagon Asset Management": {"Sector": "Generalist", "Strategy": "Event-Driven"},
    "Glazer Capital": {"Sector": "Generalist", "Strategy": "Event-Driven"},
    "Kite Lake Capital Management": {"Sector": "Generalist", "Strategy": "Event-Driven"},
    "TCI Fund Management": {"Sector": "Generalist", "Strategy": "Long-Biased"},
    "Egerton Capital": {"Sector": "Generalist", "Strategy": "Long-Biased"},
    "AKO Capital": {"Sector": "Generalist", "Strategy": "Long-Biased"},
    "Hengistbury Investment Partners": {"Sector": "Generalist", "Strategy": "Long-Biased"}
}

FUNDS = list(FUND_METADATA.keys())

# --- 1. The Instant Data Engine (Upgraded with Peer-Consensus Auto-Scaler) ---
@st.cache_data
def load_local_data():
    df = pd.read_csv('13f_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Put_Call'] = df['Put_Call'].fillna('SHARE') 
    
    # --- SMART AUTO-SCALER: Peer-Consensus Pricing ---
    # Only calculate for rows where we actually have shares and value
    mask_valid = (df['Shares'] > 0) & (df['Value ($1000s)'] > 0)
    
    # 1. Calculate the fund's implied share price (Value is in thousands, so multiply by 1000)
    df['Implied_Price'] = 0.0
    df.loc[mask_valid, 'Implied_Price'] = (df.loc[mask_valid, 'Value ($1000s)'] * 1000) / df.loc[mask_valid, 'Shares']
    
    # 2. Find the true "Consensus Price" (median) for each stock in each specific quarter
    df['Consensus_Price'] = df.groupby(['Stock', 'Date'])['Implied_Price'].transform(lambda x: x[x > 0].median())
    
    # 3. If a fund's implied price is ~1/1000th of the consensus, they reported in thousands by mistake. Fix it.
    mask_too_small = mask_valid & (df['Implied_Price'] < (df['Consensus_Price'] * 0.01))
    df.loc[mask_too_small, 'Value ($1000s)'] = df.loc[mask_too_small, 'Value ($1000s)'] * 1000
    
    # 4. If a fund's implied price is ~1000x the consensus, they reported in exact dollars by mistake. Fix it.
    mask_too_big = mask_valid & (df['Implied_Price'] > (df['Consensus_Price'] * 100))
    df.loc[mask_too_big, 'Value ($1000s)'] = df.loc[mask_too_big, 'Value ($1000s)'] / 1000
    
    # Clean up the temporary math columns so they don't clog up your tables
    df.drop(columns=['Implied_Price', 'Consensus_Price'], inplace=True, errors='ignore')

    # --- MAP SECTOR & STRATEGY FROM METADATA ---
    df['Sector'] = df['Fund'].apply(lambda x: FUND_METADATA.get(x, {}).get("Sector", "Generalist"))
    df['Strategy'] = df['Fund'].apply(lambda x: FUND_METADATA.get(x, {}).get("Strategy", "L/S Equity"))
    
    return df

try:
    df_raw = load_local_data()
except FileNotFoundError:
    st.error("Could not find '13f_data.csv'. Please run update_data.py first!")
    st.stop()

df_raw = df_raw.sort_values('Date')

# Lock in a clean, professional chart template natively
chart_template = "plotly_white"

# --- 2. Master Filters ---
st.markdown("### Master Filters")

# SINGLE MASTER CHECKBOX
master_select_all = st.checkbox("Select All (Strategies, Sectors, and Funds)", value=True)

# Strategy & Sector Level (REORDERED)
col_s1, col_s2 = st.columns(2)
with col_s1:
    all_strats = sorted(df_raw['Strategy'].unique())
    selected_strategies = st.multiselect("Filter by Strategy:", options=all_strats, default=all_strats if master_select_all else [])

with col_s2:
    all_sects = sorted(df_raw['Sector'].unique())
    selected_sectors = st.multiselect("Filter by Sector:", options=all_sects, default=all_sects if master_select_all else [])

# Cascading Fund List
filtered_funds_list = sorted(df_raw[(df_raw['Sector'].isin(selected_sectors)) & (df_raw['Strategy'].isin(selected_strategies))]['Fund'].unique().tolist())

col_f1, col_f2 = st.columns(2)

with col_f1:
    selected_funds = st.multiselect(
        "Select the funds you want to compare:", 
        options=filtered_funds_list, 
        default=filtered_funds_list if master_select_all else []
    )

with col_f2:
    available_types = df_raw['Put_Call'].unique().tolist()
    selected_types = st.multiselect(
        "Select Position Type (Shares vs. Options):", 
        options=available_types, 
        default=['SHARE'] if 'SHARE' in available_types else available_types
    )

df_filtered = df_raw[
    (df_raw['Fund'].isin(selected_funds)) & 
    (df_raw['Put_Call'].isin(selected_types)) &
    (df_raw['Sector'].isin(selected_sectors)) &
    (df_raw['Strategy'].isin(selected_strategies))
].copy()

LINE_COLORS = px.colors.qualitative.Safe

# --- 3. Dashboard Interface ---
if len(selected_funds) > 0 and len(selected_types) > 0:
    total_aum_history = df_filtered.groupby(['Date', 'Fund'])['Value ($1000s)'].sum().reset_index()
    total_aum_history.rename(columns={'Value ($1000s)': 'Total AUM'}, inplace=True)
    
    tab1, tab2, tab3 = st.tabs(["Portfolio Overlap", "Deep Dive: Single Stock", "Conviction & Holding Periods"])
    
    with tab1:
        available_dates = sorted(df_filtered['Date'].unique(), reverse=True)
        date_options = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in available_dates]
        
        st.markdown("### Point-in-Time Consensus Snapshot")
        selected_date_str = st.selectbox("Select a Quarter to Analyze:", options=date_options, index=0)
        selected_date = pd.to_datetime(selected_date_str)
        
        df_latest = df_filtered[df_filtered['Date'] == selected_date].copy()
        
        df_latest_aum = total_aum_history[total_aum_history['Date'] == selected_date][['Fund', 'Total AUM']]
        df_latest = pd.merge(df_latest, df_latest_aum, on='Fund', how='left')
        
        df_latest['Concentration %'] = 0.0
        mask = df_latest['Total AUM'] > 0
        df_latest.loc[mask, 'Concentration %'] = (df_latest.loc[mask, 'Value ($1000s)'] / df_latest.loc[mask, 'Total AUM']) * 100
        
        latest_grouped = df_latest.groupby('Stock').agg(
            Num_Funds=('Fund', 'nunique'),
            Median_Concentration=('Concentration %', 'median')
        ).reset_index()
        
        if not df_latest.empty:
            idx_min = df_latest.groupby('Stock')['Concentration %'].idxmin()
            idx_max = df_latest.groupby('Stock')['Concentration %'].idxmax()
            
            min_details = df_latest.loc[idx_min].set_index('Stock')
            max_details = df_latest.loc[idx_max].set_index('Stock')
            
            latest_grouped['Min_Fund'] = latest_grouped['Stock'].map(min_details['Fund'])
            latest_grouped['Min_Val'] = latest_grouped['Stock'].map(min_details['Concentration %'])
            
            latest_grouped['Max_Fund'] = latest_grouped['Stock'].map(max_details['Fund'])
            latest_grouped['Max_Val'] = latest_grouped['Stock'].map(max_details['Concentration %'])
            
            latest_grouped['Min Concentration'] = latest_grouped.apply(lambda x: f"{x['Min_Val']:.2f}% ({x['Min_Fund']})", axis=1)
            latest_grouped['Max Concentration'] = latest_grouped.apply(lambda x: f"{x['Max_Val']:.2f}% ({x['Max_Fund']})", axis=1)
        
        latest_overlap = latest_grouped[latest_grouped['Num_Funds'] > 1].copy()
        
        if not latest_overlap.empty:
            max_stocks = len(latest_overlap)
            
            if max_stocks > 1:
                default_stocks = min(20, max_stocks) 
                st.write("Adjust the slider to filter for the top consensus names:")
                top_n = st.slider("Number of Top Consensus Stocks to Show:", min_value=1, max_value=max_stocks, value=default_stocks)
            else:
                top_n = 1
                st.info("Only 1 overlapping position found for this filter, displaying it below.")
            
            combined_chart_data = latest_overlap.sort_values(by=['Num_Funds', 'Median_Concentration'], ascending=[False, False]).head(top_n).copy()
            
            total_selected = len(selected_funds)
            combined_chart_data['Funds_Holding'] = combined_chart_data['Num_Funds'].astype(str) + f" out of {total_selected}"
            
            fig_combined = px.scatter(
                combined_chart_data, 
                x='Stock', 
                y='Median_Concentration', 
                size='Num_Funds', 
                color='Num_Funds',
                hover_data={
                    'Num_Funds': False, 
                    'Funds_Holding': True, 
                    'Median_Concentration': ':.2f',
                    'Min Concentration': True, 
                    'Max Concentration': True  
                },
                title=f"Consensus & Conviction: Top {top_n} Shared Stocks (As of {selected_date_str})",
                labels={
                    'Median_Concentration': 'Median Concentration (%)', 
                    'Funds_Holding': 'Funds Holding'
                },
                size_max=35, 
                opacity=0.8, 
                color_continuous_scale='Viridis', 
                template=chart_template
            )
            
            fig_combined.update_xaxes(categoryorder='array', categoryarray=combined_chart_data['Stock'])
            st.plotly_chart(fig_combined, use_container_width=True)
        else:
            st.info(f"No overlapping positions found in the quarter ending {selected_date_str} for these funds/filters.")
            
        st.markdown("---")
        st.subheader("Historical Overlap Trends")
        
        if len(selected_funds) < 2:
            st.info("Please select at least 2 funds from the filter above to calculate overlap.")
        else:
            history_counts = df_filtered.groupby(['Date', 'Stock'])['Fund'].nunique().reset_index()
            history_overlap = history_counts[history_counts['Fund'] > 1]
            df_history_overlap = pd.merge(df_filtered, history_overlap[['Date', 'Stock']], on=['Date', 'Stock'], how='inner')
            
            if not df_history_overlap.empty:
                overlap_trend = df_history_overlap.groupby(['Date', 'Fund'])['Value ($1000s)'].sum().reset_index()
                
                fig_overlap = px.line(
                    overlap_trend, x='Date', y='Value ($1000s)', color='Fund', markers=True,
                    title='Total Value of Overlapping Positions Over Time ($)',
                    template=chart_template,
                    color_discrete_sequence=LINE_COLORS
                )
                st.plotly_chart(fig_overlap, use_container_width=True)
                
                overlap_pct_df = pd.merge(overlap_trend, total_aum_history, on=['Date', 'Fund'], how='left')
                
                overlap_pct_df['Overlap %'] = 0.0
                mask2 = overlap_pct_df['Total AUM'] > 0
                overlap_pct_df.loc[mask2, 'Overlap %'] = (overlap_pct_df.loc[mask2, 'Value ($1000s)'] / overlap_pct_df.loc[mask2, 'Total AUM']) * 100
                
                fig_overlap_pct = px.line(
                    overlap_pct_df, x='Date', y='Overlap %', color='Fund', markers=True,
                    title='Overall Portfolio Concentration in Shared Ideas Over Time (%)',
                    template=chart_template,
                    color_discrete_sequence=LINE_COLORS
                )
                fig_overlap_pct.update_layout(yaxis_title="Overlap (% of Total Portfolio)")
                st.plotly_chart(fig_overlap_pct, use_container_width=True)
                
            else:
                st.info("No historical overlap found between the currently selected funds.")

    with tab2:
        st.subheader("Deep Dive: Single Stock Analysis")
        
        all_unique_stocks = sorted(df_filtered['Stock'].unique())
        selected_stock = st.selectbox("Select a stock to view its history among the chosen funds:", all_unique_stocks)
        
        df_stock = df_filtered[df_filtered['Stock'] == selected_stock].copy()
        
        if not df_stock.empty:
            fig_stock_val = px.line(
                df_stock, x='Date', y='Value ($1000s)', color='Fund', markers=True,
                title=f'Absolute Holding Value ($) Over Time',
                template=chart_template,
                color_discrete_sequence=LINE_COLORS
            )
            st.plotly_chart(fig_stock_val, use_container_width=True)
            
            fig_stock_shares = px.line(
                df_stock, x='Date', y='Shares', color='Fund', markers=True,
                title=f'Share Count Over Time (True Buying/Selling)',
                template=chart_template,
                color_discrete_sequence=LINE_COLORS
            )
            st.plotly_chart(fig_stock_shares, use_container_width=True)
            
            df_stock_concentration = pd.merge(df_stock, total_aum_history, on=['Date', 'Fund'], how='left')
            
            df_stock_concentration['Concentration %'] = 0.0
            mask3 = df_stock_concentration['Total AUM'] > 0
            df_stock_concentration.loc[mask3, 'Concentration %'] = (df_stock_concentration.loc[mask3, 'Value ($1000s)'] / df_stock_concentration.loc[mask3, 'Total AUM']) * 100
            
            fig_concentration = px.line(
                df_stock_concentration, x='Date', y='Concentration %', color='Fund', markers=True,
                title=f'{selected_stock} Concentration (% of Total Portfolio) Over Time',
                template=chart_template,
                color_discrete_sequence=LINE_COLORS
            )
            fig_concentration.update_layout(yaxis_title="Concentration (%)")
            st.plotly_chart(fig_concentration, use_container_width=True)
            
            st.markdown("---")
            st.write(f"### Raw Historical Data for {selected_stock}")
            
            data_view = st.radio(
                "Select Data Metric to View in Table:",
                options=["Value ($1000s)", "Share Count"],
                horizontal=True
            )
            
            if data_view == "Value ($1000s)":
                pivot_val = df_stock.pivot_table(index='Date', columns='Fund', values='Value ($1000s)', fill_value=0)
                pivot_val.index = pivot_val.index.strftime('%Y-%m-%d')
                st.dataframe(pivot_val.sort_index(ascending=False).style.format("${:,.0f}"), use_container_width=True)
            else:
                pivot_shares = df_stock.pivot_table(index='Date', columns='Fund', values='Shares', fill_value=0)
                pivot_shares.index = pivot_shares.index.strftime('%Y-%m-%d')
                st.dataframe(pivot_shares.sort_index(ascending=False).style.format("{:,.0f}"), use_container_width=True)

    with tab3:
        st.subheader("Aggregate Conviction: Holding Periods")
        
        # Calculate holding math
        latest_dates_per_fund = df_filtered.groupby('Fund')['Date'].max().reset_index()
        latest_dates_per_fund.rename(columns={'Date': 'Fund_Latest_Date'}, inplace=True)
        
        holdings_summary = df_filtered.groupby(['Fund', 'Stock']).agg(
            First_Quarter=('Date', 'min'),
            Last_Quarter=('Date', 'max'),
            Quarters_Held=('Date', 'nunique')
        ).reset_index()
        
        holdings_summary = pd.merge(holdings_summary, latest_dates_per_fund, on='Fund', how='left')
        holdings_summary['Status'] = holdings_summary.apply(
            lambda x: 'Open' if x['Last_Quarter'] == x['Fund_Latest_Date'] else 'Closed', axis=1
        )
        
        # 1. Box plot to natively show Median, Min, Max, and spread
        fig_hold_dist = px.box(
            holdings_summary, x='Fund', y='Quarters_Held', color='Status',
            title="Holding Period Distribution (Median, Min, Max) by Fund",
            labels={'Quarters_Held': 'Quarters Held'},
            template=chart_template,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_hold_dist, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Fund Deep Dive: Holding Analysis")
        
        # 2. Individual fund selection for deep dive
        fund_to_analyze = st.selectbox("Select a specific fund to analyze their turnover:", selected_funds)
        fund_holdings = holdings_summary[holdings_summary['Fund'] == fund_to_analyze].copy()
        
        open_stats = fund_holdings[fund_holdings['Status'] == 'Open']['Quarters_Held']
        closed_stats = fund_holdings[fund_holdings['Status'] == 'Closed']['Quarters_Held']
        
        # Calculate Median, Min, and Max safely
        open_med = open_stats.median() if not open_stats.empty else 0
        open_min = open_stats.min() if not open_stats.empty else 0
        open_max = open_stats.max() if not open_stats.empty else 0
        
        closed_med = closed_stats.median() if not closed_stats.empty else 0
        closed_min = closed_stats.min() if not closed_stats.empty else 0
        closed_max = closed_stats.max() if not closed_stats.empty else 0
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Median Quarters Held (Open Positions)", f"{open_med:.0f}")
        col_m1.caption(f"**Min:** {open_min:.0f} quarter(s) | **Max:** {open_max:.0f} quarters")
        
        col_m2.metric("Median Quarters Held (Closed Positions)", f"{closed_med:.0f}")
        col_m2.caption(f"**Min:** {closed_min:.0f} quarter(s) | **Max:** {closed_max:.0f} quarters")
        
        # 3. Histogram of holding periods
        fig_hist = px.histogram(
            fund_holdings, x='Quarters_Held', color='Status', barmode='overlay',
            title=f"Distribution of Holding Periods for {fund_to_analyze}",
            labels={'Quarters_Held': 'Quarters Held'},
            template=chart_template,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # 4. Tables showing Longest Held and Quick Trades
        fund_holdings['First_Quarter'] = fund_holdings['First_Quarter'].dt.strftime('%Y-%m-%d')
        fund_holdings['Last_Quarter'] = fund_holdings['Last_Quarter'].dt.strftime('%Y-%m-%d')
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.write("**Longest Held Open Positions (Core Portfolio)**")
            open_pos = fund_holdings[fund_holdings['Status'] == 'Open'].sort_values('Quarters_Held', ascending=False).head(15)
            open_display = open_pos[['Stock', 'First_Quarter', 'Quarters_Held']].set_index('Stock')
            st.dataframe(open_display, use_container_width=True)
            
        with col_t2:
            st.write("**Quick Trades (Positions Closed in 1-2 Quarters)**")
            quick_trades = fund_holdings[(fund_holdings['Status'] == 'Closed') & (fund_holdings['Quarters_Held'] <= 2)]
            quick_trades_sorted = quick_trades.sort_values('Last_Quarter', ascending=False).head(15)
            closed_display = quick_trades_sorted[['Stock', 'First_Quarter', 'Last_Quarter', 'Quarters_Held']].set_index('Stock')
            st.dataframe(closed_display, use_container_width=True)

else:
    st.warning("Please select at least one fund and one position type to view the charts.")