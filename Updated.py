import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- SETUP ---
st.set_page_config(page_title="Financial Model Dashboard", layout="wide")

# --- Beautification ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
    background-color: #F5F7FA;
    font-size: 15px;
}
.stDataFrame, .stPlotlyChart, .stSelectbox, .stRadio, .stButton, .stSubheader {
    background-color: white;
    border-radius: 15px;
    padding: 10px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 15px;
}
h1, h2, h3, h4, h5, h6 {
    color: #222831;
}
</style>
""", unsafe_allow_html=True)

# --- Dashboard Title ---
st.markdown("""
    <h1 style='text-align: center; font-size: 30px; color: #393E46;'>📊 Financial Model & Behavior Insight Dashboard</h1>
    <p style='text-align: center; color: #6C757D; font-size:15px;'>Analyzing Market Behaviors using Technical Indicators</p>
    <hr style='margin-top:0px;'>
""", unsafe_allow_html=True)

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload your Nifty 50 dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.warning("⚠️ Please upload a dataset CSV file to continue.")
    st.stop()

# --- Load Data ---
df = pd.read_csv(uploaded_file, parse_dates=['Date'])
df.set_index('Date', inplace=True)

# --- Feature Engineering ---
try:
    df['MA_short'] = df['Price'].rolling(window=5).mean()
    df['MA_long'] = df['Price'].rolling(window=10).mean()
    df['Volatility_short'] = df['Price'].rolling(window=5).std()
    df['Volatility_long'] = df['Price'].rolling(window=10).std()
    df.dropna(inplace=True)
except Exception as e:
    st.error(f"❌ Feature engineering failed: {e}")
    st.stop()

if len(df) < 30:
    st.error("❌ Not enough rows after feature engineering. Please upload more data.")
    st.stop()

# --- Feature Selection ---
features = ['MA_short', 'MA_long', 'Volatility_short', 'Volatility_long']
X = df[features]
y = np.log1p(df['Price'])

# --- Train/Test Split ---
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = np.expm1(y_train_log)
y_test = np.expm1(y_test_log)

# --- Define Models ---
model_defs = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "KNN": KNeighborsRegressor()
}

results = []
predictions = {}

for name, model in model_defs.items():
    model.fit(X_train_scaled, y_train_log)
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    predictions[name] = y_pred

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²": round(r2_score(y_test, y_pred), 4)  # ✅ Keep 4 decimal precision
    })

# ✅ Format Results DataFrame
results_df = pd.DataFrame(results).sort_values(by="RMSE")
results_df["MAE"] = results_df["MAE"].round(2)
results_df["RMSE"] = results_df["RMSE"].round(2)

# --- Plotting Functions ---
def plot_model_metrics(df):
    melted = df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    fig = px.bar(melted, x="Metric", y="Score", color="Model", barmode="group", text_auto=".4s")
    fig.update_layout(title="Model Performance Metrics", dragmode='zoom', hovermode="closest")
    return fig

def plot_actual_vs_pred(y_true, y_pred, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', name="Predicted",
                             marker=dict(size=5, color='steelblue')))
    fig.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], y=[y_true.min(), y_true.max()],
                             mode='lines', name="Ideal", line=dict(dash='dash', color='black')))
    fig.update_layout(title=f"{model_name} Prediction", xaxis_title="Actual Price",
                      yaxis_title="Predicted Price", dragmode='zoom')
    return fig

def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    fig = px.histogram(residuals, nbins=50, title=f"{model_name} Residuals")
    fig.update_layout(xaxis_title="Residual", yaxis_title="Count", dragmode='zoom')
    return fig

# --- LAYOUT ---
row1_col1, row1_col2 = st.columns([3, 7], gap="large")

with row1_col1:
    st.subheader("📊 Model Evaluation Comparison")
    st.dataframe(results_df.reset_index(drop=True), use_container_width=True)

    fig_eval = plot_model_metrics(results_df)
    fig_eval.update_layout(height=250)
    st.plotly_chart(fig_eval, use_container_width=True, config={"scrollZoom": True})

    st.markdown("<div style='background-color:white; padding:20px; border-radius:20px;'>", unsafe_allow_html=True)
    st.markdown("#### 🔎 Model Selection Tip")
    st.markdown("""
    - **High R² + Low RMSE** → Indicates a **well-fitting model**.
    - **KNN** and **Random Forest** consistently outperform others.
    - **Linear Regression** shows **very high error**, suggesting underfitting.

    ✅ **Recommendation:** Use KNN or Random Forest for the most reliable forecasts.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    
# --- Column 2: Combined Model Comparison + Insight Box ---
with row1_col2:
    st.subheader("🧹 Model Comparison")

    comp_type = st.selectbox("Select Plot Type", [
        "Actual vs Predicted", "Residual Plots"
    ])
    base_model = "Random Forest"
    compare_model = st.selectbox("Compare With", [m for m in model_defs if m != base_model])

    # Generate base model plot
    if comp_type == "Actual vs Predicted":
        fig_base = plot_actual_vs_pred(y_test, predictions[base_model], base_model)
    elif comp_type == "Residual Plots":
        fig_base = plot_residuals(y_test, predictions[base_model], base_model)
    fig_base.update_layout(height=450)

    # Generate compare model plot
    if comp_type == "Actual vs Predicted":
        fig_compare = plot_actual_vs_pred(y_test, predictions[compare_model], compare_model)
    elif comp_type == "Residual Plots":
        fig_compare = plot_residuals(y_test, predictions[compare_model], compare_model)
    fig_compare.update_layout(height=450)

    # Side-by-side inside wider column
    compare_col1, compare_col2 = st.columns(2, gap="medium")
    with compare_col1:
        st.plotly_chart(fig_base, use_container_width=True, config={"scrollZoom": True})
    with compare_col2:
        st.plotly_chart(fig_compare, use_container_width=True, config={"scrollZoom": True})


# --- Lower Dashboard Pane: Future Price Prediction & Behavior Analysis ---

# ---------------- Ensure datetime index ----------------
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df.set_index('Date', inplace=True)
else:
    df.index = pd.to_datetime(df.index, dayfirst=True, errors='coerce')

# ---------------- Simulate Drift Function ----------------
def simulate_inputs_with_smart_drift(X_last_df, horizon, drift_strength=0.02):
    base_values = X_last_df.values[0]
    simulated_data = []

    for i in range(horizon):
        # Add small realistic variation
        noise = np.random.normal(0, drift_strength, size=base_values.shape)
        next_step = base_values * (1 + noise)
        simulated_data.append(next_step)

    return pd.DataFrame(simulated_data, columns=X_last_df.columns)

# ---------------- Behavior Classifier ----------------
def classify_behavior(price_diff, pe_diff, dy_diff):
    price_dir = 'up' if price_diff > 0 else 'down'
    pe_dir = 'up' if pe_diff > 0 else 'down'
    dy_dir = 'up' if dy_diff > 0 else 'down'

    if price_dir == 'up' and pe_dir == 'up' and dy_dir == 'down':
        return "Growth Dominant"
    elif price_dir == 'up' and pe_dir == 'down' and dy_dir == 'up':
        return "Speculative Caution"
    elif price_dir == 'down' and pe_dir == 'up' and dy_dir == 'down':
        return "Uncertain Outlook"
    elif price_dir == 'down' and pe_dir == 'down' and dy_dir == 'up':
        return "Defensive Play"
    return "Mixed Signal"

# ---------------- Forecast Helper ----------------
def predict_with_interval(model_name):
    model = model_defs[model_name]
    model.fit(X_train_scaled, y_train_log)  # Train on log-transformed y

    X_last_df = df[features].iloc[[-1]]
    future_X_unscaled = simulate_inputs_with_smart_drift(X_last_df, prediction_horizon)
    future_X_scaled = scaler.transform(future_X_unscaled)

    last_index = pd.to_datetime(df.index[-1])
    future_dates = pd.date_range(start=last_index + pd.Timedelta(days=1), periods=prediction_horizon)

    preds_log = model.predict(future_X_scaled)
    preds = np.expm1(preds_log)  # Convert back from log scale

    # Std deviation should be calculated from log residuals!
    std_dev_log = np.std(y_train_log - model.predict(X_train_scaled))
    lower = np.expm1(preds_log - 1.96 * std_dev_log)
    upper = np.expm1(preds_log + 1.96 * std_dev_log)

    return pd.Series(preds, index=future_dates), lower, upper, std_dev_log, future_X_unscaled

# --- ROW 2: Future Prediction + Behavior Plot + Column 3 Placeholder ---
row2_col1, row2_col2, row2_col3 = st.columns(3, gap="large")

# --- Column 1: Future Price Predictions ---
with row2_col1:
    st.subheader("🧾 Future Price Predictions")

    prediction_horizon = st.selectbox("Select Prediction Horizon (Days)", [7, 15, 30])
    compare_models = st.multiselect("✏️ Compare Models", list(model_defs.keys()), default=["Random Forest"])

    def model_color(model_name):
        return {
            "Random Forest": "blue",
            "KNN": "purple",
            "Decision Tree": "crimson",
        }.get(model_name, "black")

    fig = go.Figure()

    # Historical with behavior tooltip
    pe = df['P/E'].fillna(method='ffill')
    pe_dir = np.where(pe.diff().fillna(0) > 0, "↑", "↓")
    dy = df['Dividend Yield'].fillna(method='ffill')
    dy_dir = np.where(dy.diff().fillna(0) > 0, "↑", "↓")

    behavior = df.apply(lambda row: classify_behavior(
        row['Price'] - df['Price'].shift(1).loc[row.name],
        row['P/E'] - df['P/E'].shift(1).loc[row.name],
        row['Dividend Yield'] - df['Dividend Yield'].shift(1).loc[row.name]
    ), axis=1)

    customdata = np.stack([pe, pe_dir, dy, dy_dir, behavior], axis=-1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Price'],
        name="Historical Price",
        mode='lines',
        line=dict(color='gray'),
        customdata=customdata,
        hovertemplate=(
            "<b>Date:</b> %{x|%Y-%m-%d}<br>"
            "<b>Price:</b> ₹%{y:.2f}<br>"
            "<b>P/E:</b> %{customdata[0]:.2f} (%{customdata[1]})<br>"
            "<b>Dividend Yield:</b> %{customdata[2]:.2f}%% (%{customdata[3]})<br>"
            "<b>Behavior:</b> %{customdata[4]}<extra></extra>"
        )
    ))

    # Forecast lines for each selected model
    for model_name in compare_models:
        pred_series, lower_band, upper_band, _, _ = predict_with_interval(model_name)
        last_price = df['Price'].iloc[-1]

        forecast_dates = [df.index[-1]] + list(pred_series.index)
        forecast_values = [last_price] + list(pred_series.values)

        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines+markers',
            name=f"Forecast ({model_name})",
            line=dict(color=model_color(model_name)),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Predicted Price:</b> ₹%{y:.2f}<extra></extra>"
        ))

        # Confidence interval band
        fig.add_trace(go.Scatter(
            x=pred_series.index,
            y=upper_band,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=pred_series.index,
            y=lower_band,
            mode='lines',
            fill='tonexty',
            line=dict(width=0),
            fillcolor='rgba(255, 165, 0, 0.2)',
            name='95% Confidence Interval',
            showlegend=True
        ))

    # Event zones
    event_zones = [
        ("2008-09-15", "2009-06-30", "2008 Crash", "red"),
        ("2016-11-08", "2017-03-31", "Demonetisation", "orange"),
        ("2020-03-01", "2021-05-31", "COVID Crash", "purple")
    ]
    for start, end, label, color in event_zones:
        fig.add_vrect(
            x0=pd.to_datetime(start), x1=pd.to_datetime(end),
            fillcolor=color, opacity=0.15,
            annotation_text=label, annotation_position="top left", line_width=0
        )

    fig.update_layout(
        title="📉 Forecast with Confidence Interval",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date",
            range=[df.index.min(), df.index.max() + pd.Timedelta(days=prediction_horizon + 5)]
        ),
        dragmode='pan'
    )

    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
    
# --- Column 2: Market Behavior Pattern Analysis ---
with row2_col2:
    st.subheader("📊 Behavior Pattern Analysis")

    df.columns = df.columns.str.strip()
    if 'Date' not in df.columns:
        df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df.dropna(subset=['Date'], inplace=True)

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

    df['Behavior'] = "-"
    for i in range(1, len(df)):
        price_dir = movement_direction(df['Price'].iloc[i], df['Price'].iloc[i - 1])
        pe_dir = movement_direction(df['P/E'].iloc[i], df['P/E'].iloc[i - 1])
        div_dir = movement_direction(df['Dividend Yield'].iloc[i], df['Dividend Yield'].iloc[i - 1])
        df.loc[df.index[i], 'Behavior'] = classify_behavior(price_dir, pe_dir, div_dir)

    insight_option = st.selectbox("Choose a visualization", [
        "Behavior Frequency",
        "Scatter: P/E vs Dividend Yield by Behavior",
        "3D Scatter: Price vs P/E vs Dividend Yield",
        "Time Heatmap: Monthly Behavior Trends",
        "Parallel Coordinates Plot (Overall)",
        "Parallel Coordinates Plot (Event Filtered)"
    ])

    market_events = {
        "2008 Crisis": ("2008-09", "2009-06"),
        "Demonetization": ("2016-11", "2017-03"),
        "COVID-19 Crash": ("2020-02", "2021-06")
    }


    # --- Plot logic ---
    if insight_option == "Behavior Frequency":
        counts = df['Behavior'].value_counts().reset_index()
        counts.columns = ['Behavior', 'Days']
        fig = px.bar(counts, x='Behavior', y='Days', color='Behavior', text='Days')

    elif insight_option == "Scatter: P/E vs Dividend Yield by Behavior":
        fig = px.scatter(df, x='P/E', y='Dividend Yield', color='Behavior', hover_data=['Price'])
        fig.add_shape(type="rect", x0=25, x1=40, y0=0, y1=1.5, fillcolor="rgba(255,0,0,0.1)", line_width=0)
        fig.add_annotation(x=32.5, y=1.55, text="🔴 High Risk", showarrow=False)
        fig.add_shape(type="rect", x0=10, x1=20, y0=1.5, y1=2.5, fillcolor="rgba(0,255,0,0.1)", line_width=0)
        fig.add_annotation(x=15, y=2.55, text="🟢 Safe Zone", showarrow=False)
        fig.add_shape(type="rect", x0=10, x1=15, y0=0, y1=1, fillcolor="rgba(255,165,0,0.2)", line_width=0)
        fig.add_annotation(x=12.5, y=1.05, text="🟠 Value Trap", showarrow=False)

    elif insight_option == "3D Scatter: Price vs P/E vs Dividend Yield":
        fig = go.Figure()
        plot_df = df[['Price', 'P/E', 'Dividend Yield', 'Behavior']].dropna()
        color_map = {'Growth Dominant': 'green', 'Defensive Play': 'blue', 'Mixed Signal': 'gray',
                     'Speculative Caution': 'orange', 'Uncertain Outlook': 'red', '-': 'black'}

        for behavior, group in plot_df.groupby('Behavior'):
            fig.add_trace(go.Scatter3d(
                x=group['P/E'], y=group['Dividend Yield'], z=group['Price'],
                mode='markers', name=behavior,
                marker=dict(size=3, color=color_map.get(behavior, 'black')),
                text=[f"{behavior}<br>Price: {p}<br>P/E: {pe}<br>Div: {dy}" for p, pe, dy in zip(group['Price'], group['P/E'], group['Dividend Yield'])],
                hovertemplate="%{text}<extra></extra>"
            ))

        def create_cube(x0, x1, y0, y1, z0, z1, color, name):
            return go.Mesh3d(
                x=[x0, x1, x1, x0, x0, x1, x1, x0],
                y=[y0, y0, y1, y1, y0, y0, y1, y1],
                z=[z0, z0, z0, z0, z1, z1, z1, z1],
                color=color, opacity=0.15, name=name, hoverinfo='skip',
                i=[0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6],
                j=[1, 2, 3, 2, 5, 3, 6, 0, 5, 6, 6, 7],
                k=[2, 3, 0, 5, 6, 6, 7, 4, 6, 7, 7, 4],
                showscale=False
            )

        zmin, zmax = plot_df['Price'].min(), plot_df['Price'].max()
        fig.add_trace(create_cube(25, 40, 0, 1.5, zmin, zmax, 'red', "High Risk Zone"))
        fig.add_trace(create_cube(10, 20, 1.5, 2.5, zmin, zmax, 'green', "Safe Zone"))
        fig.add_trace(create_cube(10, 15, 0, 1, zmin, zmax, 'orange', "Value Trap"))

        fig.update_layout(
            scene=dict(xaxis_title="P/E", yaxis_title="Dividend Yield", zaxis_title="Price", aspectmode="cube"),
            title="🔺 3D Market Behavior Scatter with Risk Zones",
            margin=dict(l=0, r=0, t=30, b=0)
        )

    elif insight_option == "Time Heatmap: Monthly Behavior Trends":
        heatmap_df = df.copy()
        heatmap_df["YearMonth"] = heatmap_df["Date"].dt.to_period("M").astype(str)
        heatmap_counts = heatmap_df.groupby(["YearMonth", "Behavior"]).size().unstack(fill_value=0)
        desired_order = ["Growth Dominant", "Speculative Caution", "Uncertain Outlook", "Defensive Play", "Mixed Signal"]
        heatmap_counts = heatmap_counts.reindex(columns=desired_order, fill_value=0)

        fig = px.imshow(
            heatmap_counts.T,
            labels=dict(x="Month", y="Behavior", color="Days"),
            x=heatmap_counts.index,
            y=heatmap_counts.columns,
            color_continuous_scale="YlGnBu",
            aspect="auto"
        )
        fig.update_layout(shapes=[
            dict(type="rect", x0="2008-09", x1="2009-06", y0=-0.5, y1=4.5, fillcolor="red", opacity=0.15, line_width=0),
            dict(type="rect", x0="2016-11", x1="2017-03", y0=-0.5, y1=4.5, fillcolor="orange", opacity=0.15, line_width=0),
            dict(type="rect", x0="2020-02", x1="2021-06", y0=-0.5, y1=4.5, fillcolor="purple", opacity=0.15, line_width=0)
        ])
        fig.add_annotation(x="2008-09", y=4.5, text="🟥 2008 Crisis", showarrow=False, yshift=10)
        fig.add_annotation(x="2016-11", y=4.5, text="🟧 Demonetization", showarrow=False, yshift=10)
        fig.add_annotation(x="2020-02", y=4.5, text="🟪 COVID-19", showarrow=False, yshift=10)

    elif insight_option == "Parallel Coordinates Plot (Overall)":
        pc_df = df[['Price', 'P/E', 'Dividend Yield']].dropna().astype(float)
        fig = go.Figure(data=go.Parcoords(
            line=dict(color=pc_df['Price'], colorscale='Plasma', showscale=True),
            dimensions=[
                dict(label='Price', values=pc_df['Price']),
                dict(label='P/E', values=pc_df['P/E']),
                dict(label='Dividend Yield', values=pc_df['Dividend Yield'])
            ]
        ))
        fig.update_layout(margin=dict(l=50, r=50, t=30, b=30))

    elif insight_option == "Parallel Coordinates Plot (Event Filtered)":
        selected_event = st.selectbox("Select Market Event", list(market_events.keys()))
        start, end = market_events[selected_event]
        filtered_df = df[(df['Date'] >= start) & (df['Date'] <= end)][['Price', 'P/E', 'Dividend Yield']].dropna().astype(float)
        fig = go.Figure(data=go.Parcoords(
            line=dict(color=filtered_df['Price'], colorscale='Viridis', showscale=True),
            dimensions=[
                dict(label='Price', values=filtered_df['Price']),
                dict(label='P/E', values=filtered_df['P/E']),
                dict(label='Dividend Yield', values=filtered_df['Dividend Yield'])
            ]
        ))
        fig.update_layout(margin=dict(l=50, r=50, t=30, b=30))
    # --- Add Clean Dynamic Title ---
    chart_titles = {
        "Behavior Frequency": "📊 Behavior Frequency",
        "Scatter: P/E vs Dividend Yield by Behavior": "📈 P/E vs Dividend Yield Scatter",
        "3D Scatter: Price vs P/E vs Dividend Yield": "🔺 3D Market Behavior Scatter",
        "Time Heatmap: Monthly Behavior Trends": "🗓️ Monthly Behavior Heatmap",
        "Parallel Coordinates Plot (Overall)": "🔗 Parallel Coordinates Plot (Overall)",
        "Parallel Coordinates Plot (Event Filtered)": "📅 Parallel Coordinates Plot (Event Filtered)"
    }

    fig.update_layout(
        autosize=False,
        width=600,  # Fix the width so it fits nicely inside the column
        height=450,  # Optional: you can adjust height if needed
        title=dict(
            text=chart_titles.get(insight_option, "📈 Market Behavior"),
            x=0,
            xanchor='left'
        ),
        title_font=dict(size=16, color="#222831"),
        xaxis_title="",
        yaxis_title="",
        hovermode="closest",
        margin=dict(t=50),
        dragmode='pan'
    )

    # Tiny spacers (optional)
    st.markdown("")
    st.markdown("")

    # Plot the chart
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True}, height=250)


# --- Column 3: Market Insight Summary ---
with row2_col3:
    st.subheader("🧠 Market Insight Explorer")
    market_type = st.radio("Market Type", ["Normal Market", "Event Occurrence"], horizontal=True)

    # Define default full-range if Normal Market is selected
    event_options = {
        "2008 Crisis": ("2008-09-01", "2009-06-30"),
        "Demonetization": ("2016-11-01", "2017-02-01"),
        "COVID-19": ("2020-03-01", "2021-06-30")
    }

    if market_type == "Event Occurrence":
        selected_event = st.selectbox("Select Event", list(event_options.keys()))
        start, end = event_options[selected_event]
    else:
        selected_event = "-"
        start, end = df['Date'].min(), df['Date'].max()

    selected_behavior = st.selectbox("Select Market Behavior", sorted(df['Behavior'].unique()))
    current_price = st.number_input("Enter Today's Market Price", value=22547.55)
    today_date = pd.Timestamp.today()  # ⏱️ Auto fetch current date

    df_filtered = df[(df['Behavior'] == selected_behavior) &
                     (df['Date'] >= pd.to_datetime(start)) &
                     (df['Date'] <= pd.to_datetime(end))].copy()

    if not df_filtered.empty:
        df_filtered['Days'] = (today_date - df_filtered['Date']).dt.days
        df_filtered['Future_Price'] = current_price
        df_filtered['Actual_Return_%'] = ((df_filtered['Future_Price'] - df_filtered['Price']) / df_filtered['Price']) * 100
        df_filtered['CAGR_%'] = ((df_filtered['Future_Price'] / df_filtered['Price']) ** (1 / (df_filtered['Days'] / 365)) - 1) * 100
        df_sorted = df_filtered.sort_values(by='CAGR_%', ascending=False)

        # --- Summary ---
        max_cagr = df_sorted['CAGR_%'].max()
        min_cagr = df_sorted['CAGR_%'].min()
        best_return = df_sorted['Actual_Return_%'].max()
        worst_return = df_sorted['Actual_Return_%'].min()
        avg_years = df_sorted['Days'].mean() / 365
        total_records = len(df_sorted)

        st.subheader("🧾 Market Insight Summary")
        st.markdown(f"""
        <div style='font-size:11px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px;'>
            <div>📈 <b>Max CAGR</b><br>{max_cagr:.2f}%</div>
            <div>🏆 <b>Best Return</b><br>{best_return:.2f}%</div>
            <div>📉 <b>Min CAGR</b><br>{min_cagr:.2f}%</div>
            <div>💔 <b>Worst Return</b><br>{worst_return:.2f}%</div>
            <div>⏱️ <b>Avg Duration</b><br>{avg_years:.1f} yrs</div>
            <div>📊 <b>Total Records</b><br>{total_records}</div>
        </div>
        """, unsafe_allow_html=True)

        st.success(
            f"🟢 During {selected_event if selected_event != '-' else 'normal market'}, "
            f"{selected_behavior} behavior yielded up to {max_cagr:.2f}% CAGR."
        )
    else:
        st.warning("⚠️ No data found for this combination.")
        
# --- ROW 3: Full Width Summary ---
st.markdown("## 📋 Behavior Summary Insights")
summary = df.groupby('Behavior').agg(
    Price_median=('Price', 'median'),
    Price_mean=('Price', 'mean'),
    P_E_median=('P/E', 'median'),
    P_E_mean=('P/E', 'mean'),
    Dividend_Yield_median=('Dividend Yield', 'median'),
    Dividend_Yield_mean=('Dividend Yield', 'mean')
).reset_index()

def generate_insight(row):
    if row['Behavior'] == "Growth Dominant":
        return f"📈 High Price ({int(row['Price_median'])}), P/E ({row['P_E_median']:.2f}), low Div Yield ({row['Dividend_Yield_median']:.2f})"
    elif row['Behavior'] == "Defensive Play":
        return f"🛡️ Lower Price ({int(row['Price_median'])}), higher Div Yield ({row['Dividend_Yield_median']:.2f})"
    elif row['Behavior'] == "Mixed Signal":
        return "❓ Conflicting metric movements."
    elif row['Behavior'] == "Speculative Caution":
        return "⚠️ Very High Price & P/E, Low Dividend Yield. Hot market risk."
    elif row['Behavior'] == "Uncertain Outlook":
        return "🌪️ Market sentiment unclear."
    return "-"

summary['Insight'] = summary.apply(generate_insight, axis=1)
st.dataframe(summary.round(2), use_container_width=True)