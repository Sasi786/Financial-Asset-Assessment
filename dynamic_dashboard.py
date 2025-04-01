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

# App config
st.set_page_config(page_title="📈 Financial Model Dashboard", layout="wide")
st.title("📈 Financial Asset Price Prediction Dashboard")

# Load data
data_path = "Cleaned_Nifty_50_Dataset.csv"
df = pd.read_csv(data_path)
df.dropna(inplace=True)

# Define features and target
features = ['P/E', 'Div Yield %', 'P/B']
target = 'Price'
X = df[features]
y = df[target]

# Train-test split
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

# Evaluation
results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Fixed for compatibility
    r2 = r2_score(y_test, y_pred)
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2 Score": r2})

results_df = pd.DataFrame(results).sort_values(by="RMSE").reset_index(drop=True)

# Dropdown structure
view_option = st.selectbox("📌 Select View Type", ["Overall Comparison", "Actual vs Predicted", "Residual Plots", "P/E vs Price", "Dividend Yield % vs Price"])

# Show results
if view_option == "Overall Comparison":
    st.subheader("📊 Model Performance Summary")
    st.dataframe(results_df.style.format("{:.2f}"))

    st.subheader("📈 Performance Comparison Chart")
    melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=melted, x="Metric", y="Value", hue="Model", ax=ax)
    st.pyplot(fig)

else:
    model_choice = st.selectbox("Choose Model", results_df['Model'].tolist())

    if view_option == "Actual vs Predicted":
        st.subheader(f"📍 Actual vs Predicted - {model_choice}")
        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions[model_choice], alpha=0.6)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title(f"Actual vs Predicted: {model_choice}")
        st.pyplot(fig)

    elif view_option == "Residual Plots":
        st.subheader(f"📍 Residual Distribution - {model_choice}")
        residuals = y_test - predictions[model_choice]
        fig, ax = plt.subplots()
        sns.histplot(residuals, kde=True, bins=30, ax=ax)
        ax.set_xlabel("Residual Error")
        ax.set_title(f"Residuals for {model_choice}")
        st.pyplot(fig)

    elif view_option == "P/E vs Price":
        st.subheader(f"📍 P/E vs Price (Colored by Prediction: {model_choice})")
        fig, ax = plt.subplots()
        ax.scatter(df['P/E'], df['Price'], alpha=0.6, label="Actual")
        ax.set_xlabel("P/E Ratio")
        ax.set_ylabel("Price")
        st.pyplot(fig)

    elif view_option == "Dividend Yield % vs Price":
        st.subheader(f"📍 Dividend Yield % vs Price (Colored by Prediction: {model_choice})")
        fig, ax = plt.subplots()
        ax.scatter(df['Div Yield %'], df['Price'], alpha=0.6, label="Actual")
        ax.set_xlabel("Dividend Yield %")
        ax.set_ylabel("Price")
        st.pyplot(fig)

# Optional: Hide sidebar
hide_sidebar = """
    <style>
        [data-testid="stSidebar"] {display: none;}
    </style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)
