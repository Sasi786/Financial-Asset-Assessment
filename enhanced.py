# --- Streamlit Dashboard for Financial Model Comparison ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Load Dataset ---
st.set_page_config(page_title="Financial Model Comparison Dashboard", layout="wide")
st.title("📊 Financial Model Comparison Dashboard")

github_url = "https://raw.githubusercontent.com/Sasi786/Financial-Asset-Assessment/main/Cleaned_Nifty_50_Dataset.csv"
df = pd.read_csv(github_url, parse_dates=['Date'], index_col='Date')

# Standardize Dividend Yield column name
if "Dividend Yeild" in df.columns:
    df.rename(columns={"Dividend Yeild": "Dividend Yield"}, inplace=True)
elif "Div Yield %" in df.columns:
    df.rename(columns={"Div Yield %": "Dividend Yield"}, inplace=True)

# --- Features and Labels ---
features = ['P/E', 'Dividend Yield', 'P/B']
target = 'Price'
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model Definitions ---
model_defs = {
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest": lambda: RandomForestRegressor(random_state=42),
    "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=42),
    "Decision Tree": lambda: DecisionTreeRegressor(random_state=42),
    "KNN": lambda: KNeighborsRegressor()
}

# --- Train and Collect Results ---
results = []
predictions = {}
for name, constructor in model_defs.items():
    model = constructor()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²": r2_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results).sort_values(by="RMSE").round(2)
tabs = st.tabs([
    "Overall Comparison", 
    "Model Prediction", 
    "Model Comparison", 
    "Future Price Prediction", 
    "Behavior Analysis"
])

# --- Tab 1 ---
with tabs[0]:
    st.subheader("📋 Model Performance Metrics")
    st.dataframe(results_df, use_container_width=True)

    st.subheader("📊 Performance Comparison")
    metrics_df = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=metrics_df, x="Metric", y="Score", hue="Model", ax=ax, palette="Set2")
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, color='black', xytext=(0, 5), textcoords='offset points')
    st.pyplot(fig)

# --- Tab 2 ---
with tabs[1]:
    st.subheader("🧪 Individual Model Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        plot_type = st.selectbox("Choose Plot", [
            "Actual vs Predicted", "Residual Plots", "Price vs P/E", "Price vs Dividend Yield"
        ], key="ind_plot")
    with col2:
        selected_model = st.selectbox("Select Model", list(model_defs.keys()), key="ind_model")

    if plot_type == "Actual vs Predicted":
        y_pred = predictions[selected_model]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred, alpha=0.6, c='teal')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title(f"Actual vs Predicted: {selected_model}")
        st.pyplot(fig)

    elif plot_type == "Residual Plots":
        residuals = y_test - predictions[selected_model]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(residuals, kde=True, bins=30, ax=ax, color="indianred")
        ax.set_title(f"Residuals for {selected_model}")
        ax.set_xlabel("Error")
        st.pyplot(fig)

    elif plot_type == "Price vs P/E":
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(x=df['P/E'], y=df['Price'], color='blueviolet', ax=ax)
        ax.set_title("P/E vs Price")
        ax.set_xlabel("P/E Ratio")
        ax.set_ylabel("Price")
        st.pyplot(fig)

    elif plot_type == "Price vs Dividend Yield":
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(x=df['Div Yield %'], y=df['Price'], color='darkorange', ax=ax)
        ax.set_title("Dividend Yield % vs Price")
        ax.set_xlabel("Dividend Yield (%)")
        ax.set_ylabel("Price")
        st.pyplot(fig)

# --- Tab 3 ---
with tabs[2]:
    st.subheader("🔁 Compare Two Models")
    base_model = "Random Forest"
    comparison_type = st.selectbox("Comparison Type", [
        "Actual vs Predicted", "Residual Plots", "Price vs P/E", "Price vs Dividend Yield"
    ])
    compare_model = st.selectbox("Compare Against", [m for m in model_defs if m != base_model])

    col1, col2 = st.columns(2)
    if comparison_type == "Actual vs Predicted":
        with col1:
            fig, ax = plt.subplots()
            ax.scatter(y_test, predictions[base_model], alpha=0.5, color='mediumseagreen')
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
            ax.set_title(base_model)
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            ax.scatter(y_test, predictions[compare_model], alpha=0.5, color='royalblue')
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
            ax.set_title(compare_model)
            st.pyplot(fig)

    elif comparison_type == "Residual Plots":
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(y_test - predictions[base_model], kde=True, bins=30, ax=ax, color="darkcyan")
            ax.set_title(base_model)
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.histplot(y_test - predictions[compare_model], kde=True, bins=30, ax=ax, color="crimson")
            ax.set_title(compare_model)
            st.pyplot(fig)

    elif comparison_type == "Price vs P/E":
        with col1:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df['P/E'], y=df['Price'], color='purple', ax=ax)
            ax.set_title(base_model)
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df['P/E'], y=df['Price'], color='green', ax=ax)
            ax.set_title(compare_model)
            st.pyplot(fig)

    elif comparison_type == "Price vs Dividend Yield":
        with col1:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df['Dividend Yield'], y=df['Price'], color='goldenrod', ax=ax)
            ax.set_title(base_model)
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df['Dividend Yield'], y=df['Price'], color='deepskyblue', ax=ax)
            ax.set_title(compare_model)
            st.pyplot(fig)

# --- Forecast Helper ---
def predict_with_interval(model_name):
    model = model_defs[model_name]()
    model.fit(X_train_scaled, y_train)

    X_last_df = df[features].iloc[[-1]]
    future_X_unscaled = simulate_inputs_with_drift(X_last_df, prediction_horizon)
    future_X_scaled = scaler.transform(future_X_unscaled)

    # ✅ Fix: Ensure index is datetime
    last_index = pd.to_datetime(df.index[-1])
    future_dates = pd.date_range(start=last_index + pd.Timedelta(days=1), periods=prediction_horizon)

    preds = model.predict(future_X_scaled)
    std_dev = np.std(y_train - model.predict(X_train_scaled))
    lower = preds - 1.96 * std_dev
    upper = preds + 1.96 * std_dev

    return pd.Series(preds, index=future_dates), lower, upper, std_dev, future_X_unscaled

# Ensure datetime index for proper plotting
df.index = pd.to_datetime(df.index, dayfirst=True) 

# --- Tab 4: Future Price Prediction ---
with tabs[3]:
    st.subheader("📈 Future Price Predictions")

    prediction_horizon = st.selectbox("Select Prediction Horizon (Days)", [7, 15, 30], key="pred_horizon")
    compare_models = st.multiselect("🧪 Compare Models", list(model_defs.keys()), default=["Random Forest"])

    def model_color(model_name):
        return {
            "Random Forest": "blue",
            "Gradient Boosting": "orange",
            "KNN": "purple",
            "Decision Tree": "crimson",
            "Linear Regression": "teal"
        }.get(model_name, "black")

    def simulate_inputs_with_drift(X_last_df, horizon):
        drift = np.array([0.1, -0.01, 0.05])
        return pd.DataFrame([X_last_df.values[0] + i * drift for i in range(horizon)], columns=features)

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
        else:
            return "Mixed Signal"

    def predict_with_interval(model_name):
        model = model_defs[model_name]()
        model.fit(X_train_scaled, y_train)
        X_last_df = df[features].iloc[[-1]]
        future_X_unscaled = simulate_inputs_with_drift(X_last_df, prediction_horizon)
        future_X_scaled = scaler.transform(future_X_unscaled)
        last_date = pd.to_datetime(df.index[-1])
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_horizon)
        preds = model.predict(future_X_scaled)
        std_dev = np.std(y_train - model.predict(X_train_scaled))
        lower = preds - 1.96 * std_dev
        upper = preds + 1.96 * std_dev
        return pd.Series(preds, index=future_dates), lower, upper, std_dev, future_X_unscaled

    fig = go.Figure()

    # Historical line with behavior tooltip
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Price'],
        name="Historical Price",
        mode='lines',
        line=dict(color='gray'),
        customdata=np.stack([
            df['P/E'],
            np.where(df['P/E'].diff().fillna(0) > 0, "↑", "↓"),
            df['Dividend Yield'],
            np.where(df['Dividend Yield'].diff().fillna(0) > 0, "↑", "↓"),
            df.apply(lambda row: classify_behavior(
                row['Price'] - df['Price'].shift(1).loc[row.name],
                row['P/E'] - df['P/E'].shift(1).loc[row.name],
                row['Dividend Yield'] - df['Dividend Yield'].shift(1).loc[row.name]
            ), axis=1)
        ], axis=-1),
        hovertemplate=(
            "<b>Date:</b> %{x|%Y-%m-%d}<br>"
            "<b>Price:</b> ₹%{y:.2f}<br>"
            "<b>P/E:</b> %{customdata[0]:.2f} (%{customdata[1]})<br>"
            "<b>Dividend Yield:</b> %{customdata[2]:.2f}%% (%{customdata[3]})<br>"
            "<b>Behavior:</b> %{customdata[4]}<extra></extra>"
        )
    ))

    for model_name in compare_models:
        pred_series, lower_band, upper_band, _, future_df = predict_with_interval(model_name)
        last_price = df['Price'].iloc[-1]
        forecast_dates = [df.index[-1]] + list(pred_series.index)
        forecast_values = [last_price] + list(pred_series.values)

        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines+markers',
            name=f"Forecast ({model_name})",
            line=dict(color=model_color(model_name), dash="solid"),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Predicted Price:</b> ₹%{y:.2f}<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=pred_series.index, y=upper_band,
            mode='lines', line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=pred_series.index, y=lower_band,
            mode='lines', fill='tonexty', line=dict(width=0),
            fillcolor='rgba(255, 165, 0, 0.2)', name='95% Confidence Interval', showlegend=True
        ))

    # Highlight major event periods
    event_zones = [
        ("2008-09-15", "2009-06-30", "2008 Crash", "red"),
        ("2016-11-08", "2017-03-31", "Demonetisation", "orange"),
        ("2020-03-01", "2021-05-31", "COVID Crash", "purple")
    ]
    for start, end, label, color in event_zones:
        fig.add_vrect(
            x0=start, x1=end, fillcolor=color, opacity=0.15,
            annotation_text=label, annotation_position="top left", line_width=0
        )

    fig.update_layout(
        title="Forecast with Confidence Interval",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date",
            range=[df.index.min(), df.index.max() + pd.Timedelta(days=prediction_horizon)]
        ),
        dragmode='pan'
    )

    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    st.markdown("### 📉 Model Comparison Summary")
    summary_rows = []
    for model_name in compare_models:
        y_pred = model_defs[model_name]().fit(X_train_scaled, y_train).predict(X_test_scaled)
        summary_rows.append({
            "Model": model_name,
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R²": r2_score(y_test, y_pred)
        })
    st.dataframe(pd.DataFrame(summary_rows).round(2), use_container_width=True)
    
from plotly.graph_objs import Figure, Parcoords  # For custom parallel plots

def draw_zone_cube(fig, x_range, y_range, z_range, color, name):
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range

    fig.add_trace(go.Mesh3d(
        x=[x0, x1, x1, x0, x0, x1, x1, x0],
        y=[y0, y0, y1, y1, y0, y0, y1, y1],
        z=[z0, z0, z0, z0, z1, z1, z1, z1],
        i=[0, 0, 0, 4, 4, 5, 1, 2, 3, 7, 6, 6],
        j=[1, 2, 3, 5, 6, 6, 5, 3, 0, 6, 7, 2],
        k=[2, 3, 0, 6, 7, 7, 1, 0, 1, 2, 3, 5],
        opacity=0.15,
        color=color,
        name=name,
        showscale=False,
        hoverinfo='skip'
    ))
    
# --- Tab 5 : Market Behavior Pattern Analysis ---
with tabs[4]:
    st.subheader("📊 Market Behavior Pattern Analysis")

    df.columns = df.columns.str.strip()

    if 'Date' not in df.columns:
        if df.index.name and 'date' in df.index.name.lower():
            df['Date'] = pd.to_datetime(df.index, errors='coerce', dayfirst=True)
        elif 'IndexName' in df.columns:
            df['Date'] = pd.to_datetime(df['IndexName'], errors='coerce', dayfirst=True)
        else:
            st.error("❌ 'Date' column not found.")
            st.stop()
    else:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)

    df.dropna(subset=['Date'], inplace=True)

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

    df['Behavior'] = "-"
    for i in range(1, len(df)):
        price_dir = movement_direction(df['Price'].iloc[i], df['Price'].iloc[i - 1])
        pe_dir = movement_direction(df['P/E'].iloc[i], df['P/E'].iloc[i - 1])
        div_dir = movement_direction(df['Dividend Yield'].iloc[i], df['Dividend Yield'].iloc[i - 1])
        df.loc[df.index[i], 'Behavior'] = classify_behavior(price_dir, pe_dir, div_dir)

    # --- Visualization Selector ---
    st.markdown("### 📌 Select Insight Plot:")
    insight_option = st.selectbox("Choose a visualization", [
        "Behavior Frequency",
        "Scatter: P/E vs Dividend Yield by Behavior",
        "3D Scatter: Price vs P/E vs Dividend Yield",
        "Time Heatmap: Monthly Behavior Trends",
        "Parallel Coordinates Plot (Overall)",
        "Parallel Coordinates Plot (Event Filtered)"
    ])

    market_events = {
        "2008 Crisis": ("2008-09-15", "2009-06-30"),
        "Demonetization": ("2016-11-08", "2017-03-31"),
        "COVID-19 Crash": ("2020-02-01", "2021-06-30")
    }

    # --- Behavior Frequency ---
    if insight_option == "Behavior Frequency":
        counts = df['Behavior'].value_counts().reset_index()
        counts.columns = ['Behavior', 'Days']
        fig = px.bar(counts, x='Behavior', y='Days', color='Behavior', text='Days',
                     title='🧠 Frequency of Market Behavior Patterns')
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    # --- 2D Scatter Plot ---
    elif insight_option == "Scatter: P/E vs Dividend Yield by Behavior":
        fig = px.scatter(df, x='P/E', y='Dividend Yield', color='Behavior',
                         hover_data=['Price'], title="📉 P/E vs Dividend Yield by Behavior")
        fig.add_shape(type="rect", x0=25, x1=40, y0=0, y1=1.5, fillcolor="rgba(255,0,0,0.1)", line_width=0)
        fig.add_annotation(x=32.5, y=1.5, text="🔴 High Risk Zone", showarrow=False)
        fig.add_shape(type="rect", x0=10, x1=20, y0=1.5, y1=2.5, fillcolor="rgba(0,255,0,0.1)", line_width=0)
        fig.add_annotation(x=15, y=2.5, text="🟢 Safe Zone", showarrow=False)
        fig.add_shape(type="rect", x0=10, x1=15, y0=0, y1=1, fillcolor="rgba(255,165,0,0.15)", line_width=0)
        fig.add_annotation(x=12.5, y=1, text="🟠 Value Trap", showarrow=False)
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    # --- 3D Scatter Plot ---
    elif insight_option == "3D Scatter: Price vs P/E vs Dividend Yield":
        st.subheader("🔺 3D Behavior Plot")
        plot_df = df[['Price', 'P/E', 'Dividend Yield', 'Behavior']].dropna()
        fig = go.Figure()
        behavior_colors = {
            'Growth Dominant': 'red', 'Defensive Play': 'green', 'Mixed Signal': 'gray',
            'Speculative Caution': 'orange', 'Uncertain Outlook': 'blue', '-': 'black'
        }

        for behavior, group in plot_df.groupby('Behavior'):
            fig.add_trace(go.Scatter3d(
                x=group['P/E'], y=group['Dividend Yield'], z=group['Price'],
                mode='markers', name=behavior,
                marker=dict(size=3, color=behavior_colors.get(behavior, 'black')),
                text=[f"{behavior}<br>Price: {p}<br>P/E: {pe}<br>Div: {dy}"
                      for p, pe, dy in zip(group['Price'], group['P/E'], group['Dividend Yield'])],
                hovertemplate="%{text}<extra></extra>"
            ))

        zmin, zmax = plot_df['Price'].min(), plot_df['Price'].max()
        draw_zone_cube(fig, [25, 40], [0, 1.5], [zmin, zmax], 'red', 'High Risk')
        draw_zone_cube(fig, [10, 20], [1.5, 2.5], [zmin, zmax], 'green', 'Safe')
        draw_zone_cube(fig, [10, 15], [0, 1], [zmin, zmax], 'orange', 'Value Trap')

        fig.update_layout(scene=dict(xaxis_title="P/E", yaxis_title="Dividend Yield", zaxis_title="Price"),
                          dragmode="orbit")
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    # --- Time Heatmap: With Shaded Market Events ---
    elif insight_option == "Time Heatmap: Monthly Behavior Trends":
        st.subheader("📅 Monthly Behavior Trend Heatmap")

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
            aspect="auto",
            title="🗓️ Monthly Frequency of Market Behavior Patterns"
        )

        # Highlight event periods
        fig.update_layout(shapes=[
            dict(type="rect", x0="2008-09", x1="2009-06", y0=-0.5, y1=4.5, fillcolor="red", opacity=0.15, line_width=0),
            dict(type="rect", x0="2016-11", x1="2017-03", y0=-0.5, y1=4.5, fillcolor="orange", opacity=0.15, line_width=0),
            dict(type="rect", x0="2020-02", x1="2021-06", y0=-0.5, y1=4.5, fillcolor="purple", opacity=0.15, line_width=0)
        ])

        # Add annotations for event names
        fig.add_annotation(x="2008-09", y=4.5, text="🟥 2008 Crisis", showarrow=False, yshift=10)
        fig.add_annotation(x="2016-11", y=4.5, text="🟧 Demonetization", showarrow=False, yshift=10)
        fig.add_annotation(x="2020-02", y=4.5, text="🟪 COVID-19", showarrow=False, yshift=10)

        fig.update_layout(
            xaxis_nticks=20,
            yaxis=dict(tickmode='linear'),
            hovermode="closest",
            margin=dict(l=40, r=40, t=80, b=40)
        )
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    # --- Parallel Coordinates (Overall) ---
    elif insight_option == "Parallel Coordinates Plot (Overall)":
        st.subheader("🧭 Parallel Coordinates Plot – Overall")
        pc_df = df[['Price', 'P/E', 'Dividend Yield']].dropna().astype(float)
        fig = go.Figure(data=go.Parcoords(
            line=dict(color=pc_df['Price'], colorscale='Plasma', showscale=True),
            dimensions=[
                dict(label='Price', values=pc_df['Price']),
                dict(label='P/E', values=pc_df['P/E']),
                dict(label='Dividend Yield', values=pc_df['Dividend Yield'])
            ]
        ))
        fig.update_layout(margin=dict(l=100, r=100, t=50, b=50), autosize=True)
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    # --- Parallel Coordinates (Filtered) ---
    elif insight_option == "Parallel Coordinates Plot (Event Filtered)":
        st.subheader("🎯 Parallel Coordinates – Filtered by Event")
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
        fig.update_layout(margin=dict(l=100, r=100, t=50, b=50), autosize=True)
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    # --- Summary Table ---
    st.markdown("### 📋 Behavior Summary Insights")
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
        else:
            return "-"

    summary['Insight'] = summary.apply(generate_insight, axis=1)
    st.dataframe(summary.round(2), use_container_width=True)
