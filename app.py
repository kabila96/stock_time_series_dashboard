# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px


# ----- Page config (do this ONCE and first) -----
st.set_page_config(page_title="Tech Stocks Analysis Dashboard", layout="wide")

# ----- Title -----
st.title("POWELL NDLOVU THE ENVIRONMENTAL DATA ANALYST PORTFOLIO")
st.title("📊 Tech Stocks Analysis Dashboard")
st.caption("AAPL • AMZN • GOOG • MSFT")

# ----- File list -----
company_list = [
    r'AAPL_data.csv',
    r'AMZN_data.csv',
    r'GOOG_data.csv',
    r'MSFT_data.csv',
]

# ----- Load & combine safely -----
all_data = pd.DataFrame()
problems = []

for file in company_list:
    p = Path(file)
    if not p.exists():
        problems.append(f"Missing file: {file}")
        continue

    try:
        df = pd.read_csv(p)
        # If Name column is missing, infer from filename prefix (AAPL, AMZN, etc.)
        if 'Name' not in df.columns:
            df['Name'] = p.stem.split('_')[0]  # e.g., AAPL from AAPL_data
        # Ensure required columns exist
        needed = {'date','open','high','low','close','volume','Name'}
        missing = needed - set(df.columns)
        if missing:
            problems.append(f"{p.name} missing columns: {', '.join(sorted(missing))}")
            continue
        all_data = pd.concat([all_data, df], ignore_index=True)
    except Exception as e:
        problems.append(f"Error reading {file}: {e}")

if all_data.empty:
    st.error("No data loaded. Please check file paths and formats.")
    if problems:
        with st.expander("Load errors"):
            for msg in problems:
                st.write("- " + msg)
    st.stop()

# ----- Types -----
all_data['date'] = pd.to_datetime(all_data['date'])

# ----- Sidebar filters -----
st.sidebar.header("Filters")
companies = sorted(all_data['Name'].unique().tolist())
selected_company = st.sidebar.selectbox("Choose a company", companies, index=0)

metrics = ['close', 'open', 'high', 'low', 'volume']
selected_metric = st.sidebar.selectbox("Metric", metrics, index=0)

# Date range slider
min_d, max_d = all_data['date'].min(), all_data['date'].max()
start_d, end_d = st.sidebar.slider(
    "Date range",
    min_value=min_d.to_pydatetime(),
    max_value=max_d.to_pydatetime(),
    value=(min_d.to_pydatetime(), max_d.to_pydatetime())
)

# Moving averages
ma_1 = st.sidebar.number_input("Short MA (days)", min_value=2, max_value=120, value=20, step=1)
ma_2 = st.sidebar.number_input("Long MA (days)", min_value=5, max_value=250, value=50, step=1)

# ----- Filtered subset -----
dfc = all_data.query("Name == @selected_company").copy()
dfc = dfc[(dfc['date'] >= pd.to_datetime(start_d)) & (dfc['date'] <= pd.to_datetime(end_d))].sort_values('date')

# Compute MAs only for price metrics
if selected_metric != 'volume':
    dfc[f'MA_{ma_1}'] = dfc[selected_metric].rolling(ma_1).mean()
    dfc[f'MA_{ma_2}'] = dfc[selected_metric].rolling(ma_2).mean()

# ----- Layout -----
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📈 {selected_company} — {selected_metric.capitalize()} over time")

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.lineplot(data=dfc, x='date', y=selected_metric, ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel(selected_metric.capitalize())
    ax.set_title(f"{selected_company} {selected_metric.capitalize()}")

    # Overlay MAs for price metrics
    if selected_metric != 'volume':
        if dfc[f'MA_{ma_1}'].notna().any():
            sns.lineplot(data=dfc, x='date', y=f'MA_{ma_1}', ax=ax, linestyle='--')
        if dfc[f'MA_{ma_2}'].notna().any():
            sns.lineplot(data=dfc, x='date', y=f'MA_{ma_2}', ax=ax, linestyle='--')

    st.pyplot(fig, clear_figure=True)

    # Volume chart (always handy)
    st.subheader("🔊 Volume")
    fig_vol, axv = plt.subplots(figsize=(11, 2.8))
    sns.lineplot(data=dfc, x='date', y='volume', ax=axv)
    axv.set_xlabel("Date")
    axv.set_ylabel("Volume")
    axv.set_title(f"{selected_company} Volume")
    st.pyplot(fig_vol, clear_figure=True)

with col2:
    st.subheader("📊 Summary")
    # Show basic stats for price metrics
    show_cols = ['open','high','low','close','volume']
    st.dataframe(dfc[show_cols].describe().T)

    st.markdown("---")
st.subheader("📉 Correlations (interactive)")

# Let users choose correlation method
method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"], index=0)

show_cols = ['open','high','low','close','volume']
corr = dfc[show_cols].corr(method=method, numeric_only=True).round(2)

fig_c = px.imshow(
    corr,
    text_auto=True,          # show numbers in cells
    aspect="auto",
    zmin=-1, zmax=1,         # consistent scale
    color_continuous_midpoint=0
)
fig_c.update_layout(
    width=500, height=500,
    margin=dict(l=10, r=10, t=40, b=10),
    title=f"Correlation matrix ({method.title()})"
)
fig_c.update_traces(
    hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>corr: %{z}<extra></extra>"
)

st.plotly_chart(fig_c, use_container_width=True)


# ----- Any load warnings -----
if problems:
    with st.expander("⚠️ Data loading notes"):
        for msg in problems:
            st.write("- " + msg)



