import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# --- Load Data ---
github_url = "https://raw.githubusercontent.com/Sasi786/Financial-Asset-Assessment/main/Cleaned_Nifty_50_Dataset.csv"
df = pd.read_csv(github_url)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True)

# --- Behavior Classification ---
def movement_direction(val, ref):
    if val > ref: return "up"
    elif val < ref: return "down"
    return "neutral"

def classify_behavior(price_dir, pe_dir, div_dir):
    if price_dir == "up" and pe_dir == "up" and div_dir == "down":
        return "Growth Dominant"
    elif price_dir == "up" and pe_dir == "down" and div_dir == "up":
        return "Speculative Caution"
    elif price_dir == "down" and pe_dir == "up" and div_dir == "down":
        return "Uncertain Outlook"
    elif price_dir == "down" and pe_dir == "down" and div_dir == "up":
        return "Defensive Play"
    return "Mixed Signal"

df = df.sort_values('Date')
df['Behavior'] = '-'
for i in range(1, len(df)):
    price_dir = movement_direction(df['Price'].iloc[i], df['Price'].iloc[i - 1])
    pe_dir = movement_direction(df['P/E'].iloc[i], df['P/E'].iloc[i - 1])
    div_dir = movement_direction(df['Dividend Yield'].iloc[i], df['Dividend Yield'].iloc[i - 1])
    df.loc[df.index[i], 'Behavior'] = classify_behavior(price_dir, pe_dir, div_dir)

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.sidebar.header("📁 Market Scenario Selector")
market_type = st.sidebar.radio("Market Type", ["Normal Market", "Event Occurrence"])
event_options = {
    "-": (df['Date'].min(), df['Date'].max()),
    "2008 Crisis": ("2008-09-01", "2009-06-30"),
    "Demonetization": ("2016-11-01", "2017-02-01"),
    "COVID-19": ("2020-03-01", "2021-06-30")
}
selected_event = st.sidebar.selectbox("Select Event", list(event_options.keys()))
selected_behavior = st.sidebar.selectbox("Select Market Behavior", sorted(df['Behavior'].unique()))
current_price = st.sidebar.number_input("Enter Today's Price", value=22547.55)
today_date = pd.to_datetime(st.sidebar.text_input("Select Current Date", value=datetime.today().strftime('%Y-%m-%d')))

# --- Filtered Data ---
start, end = event_options[selected_event]
df_filtered = df[(df['Behavior'] == selected_behavior) &
                 (df['Date'] >= pd.to_datetime(start)) &
                 (df['Date'] <= pd.to_datetime(end))].copy()

df_filtered['Days'] = (today_date - df_filtered['Date']).dt.days
df_filtered['Future_Price'] = current_price
df_filtered['Actual_Return_%'] = ((df_filtered['Future_Price'] - df_filtered['Price']) / df_filtered['Price']) * 100
df_filtered['CAGR_%'] = ((df_filtered['Future_Price'] / df_filtered['Price']) ** (1 / (df_filtered['Days'] / 365)) - 1) * 100
df_filtered_sorted = df_filtered.sort_values(by='CAGR_%', ascending=False)

# --- Handle Empty Data ---
if df_filtered_sorted.empty:
    st.warning("⚠️ No data available for this behavior and event combination.")
    st.stop()

# --- KPI Metrics ---
st.title("📊 Behavioral Market Return Explorer")
avg_days = df_filtered_sorted['Days'].mean()
avg_years = round(avg_days / 365, 1)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("🏆 Max CAGR", f"{df_filtered_sorted['CAGR_%'].max():.2f}%")
col2.metric("📉 Min CAGR", f"{df_filtered_sorted['CAGR_%'].min():.2f}%")
col3.metric("📅 Avg Duration", f"{avg_years} yrs ({int(avg_days)}d)")
col4.metric("📈 Best Return %", f"{df_filtered_sorted['Actual_Return_%'].max():.2f}%")
col5.metric("📉 Worst Return %", f"{df_filtered_sorted['Actual_Return_%'].min():.2f}%")
col6.metric("🧪 Total Records", f"{len(df_filtered_sorted)}")

# --- Narration ---
st.markdown(
    f"""
    ### 🗣️ Market Insight
    > During **{selected_event if selected_event != "-" else "normal market"}**, **{selected_behavior}** behavior yielded up to 
    **{df_filtered_sorted['CAGR_%'].max():.2f}% CAGR** over an average duration of **{avg_years} years** (**{int(avg_days)} days**).
    """
)

# --- Top & Worst Performers ---
st.markdown(f"### 🏆 Top 3 Best Days for {selected_behavior}")
st.dataframe(df_filtered_sorted.head(3)[['Date', 'Price', 'Future_Price', 'Days', 'Actual_Return_%', 'CAGR_%']], use_container_width=True)

st.markdown(f"### 💥 Worst 3 Days for {selected_behavior}")
st.dataframe(df_filtered_sorted.tail(3)[['Date', 'Price', 'Future_Price', 'Days', 'Actual_Return_%', 'CAGR_%']], use_container_width=True)

# --- Scatter View ---
st.markdown(f"### 🔍 Scatter View for {selected_behavior} in {selected_event or 'Normal Market'}")
fig = px.scatter(df_filtered_sorted, x='Date', y='CAGR_%', color='CAGR_%',
                 color_continuous_scale='plasma', size_max=10,
                 labels={'CAGR_%': 'CAGR (%)'},
                 title=f"CAGR% Over Time for {selected_behavior}")
fig.update_traces(marker=dict(size=8, opacity=0.7))
st.plotly_chart(fig, use_container_width=True)

# --- Download CSV ---
csv_data = df_filtered_sorted.to_csv(index=False).encode('utf-8')
file_label = f"{selected_behavior.replace(' ', '_')}_{selected_event.replace(' ', '_')}_ReturnAnalysis.csv"
st.download_button("⬇️ Download Top & Bottom Cases", data=csv_data, file_name=file_label, mime="text/csv")
