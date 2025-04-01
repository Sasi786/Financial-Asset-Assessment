import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page config
st.set_page_config(page_title="Financial Model Dashboard", layout="wide")
st.title("📊 Financial Model Comparison Dashboard")

# Load data
data_path = "Cleaned_Nifty_50_Dataset.csv"
df = pd.read_csv(data_path)
df.dropna(inplace=True)

# Features and target
features = ['P/E', 'Div Yield %', 'P/B']
target = 'Price'
X = df[features]
y = df[target]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "KNN": KNeighborsRegressor()
}

# Evaluate models
results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    results.append({"Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2 Score": round(r2, 2)})

results_df = pd.DataFrame(results).sort_values(by="RMSE").reset_index(drop=True)

# Section selection
section = st.selectbox("Choose Section", ["Overall Comparison", "Actual vs Predicted", "Residual Plots", "P/E vs Price", "Dividend Yield vs Price"])

# --- Overall comparison ---
if section == "Overall Comparison":
    st.subheader("✅ Model Performance Metrics")
    st.dataframe(results_df)

    st.subheader("📊 Bar Chart Comparison")
    fig, ax = plt.subplots(figsize=(10, 6))
    melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    sns.barplot(data=melted, x="Metric", y="Value", hue="Model", ax=ax)
    st.pyplot(fig)

# --- Model-specific section ---
elif section in ["Actual vs Predicted", "Residual Plots"]:
    selected_model = st.selectbox("Choose Model", list(predictions.keys()))

    if section == "Actual vs Predicted":
        st.subheader(f"📍 Actual vs Predicted - {selected_model}")
        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions[selected_model], alpha=0.6)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        st.pyplot(fig)

    elif section == "Residual Plots":
        st.subheader(f"📉 Residual Distribution - {selected_model}")
        residuals = y_test - predictions[selected_model]
        fig, ax = plt.subplots()
        sns.histplot(residuals, kde=True, bins=30, ax=ax)
        ax.set_title(f"Residuals for {selected_model}")
        st.pyplot(fig)

# --- Feature vs Price plots ---
elif section == "P/E vs Price":
    st.subheader("P/E Ratio vs Price")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df['P/E'], y=df['Price'], color='green', ax=ax)
    ax.set_xlabel("P/E")
    ax.set_ylabel("Price")
    st.pyplot(fig)

elif section == "Dividend Yield vs Price":
    st.subheader("Dividend Yield % vs Price")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df['Div Yield %'], y=df['Price'], color='purple', ax=ax)
    ax.set_xlabel("Div Yield %")
    ax.set_ylabel("Price")
    st.pyplot(fig)

# Optional: Hide Streamlit default elements
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    </style>
""", unsafe_allow_html=True)
