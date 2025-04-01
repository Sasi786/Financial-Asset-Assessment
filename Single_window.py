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

# Page configuration
st.set_page_config(page_title="Financial Model Dashboard", layout="wide")

# Title
st.markdown("<h1 style='text-align: center;'>📊 Financial Model Comparison Dashboard</h1>", unsafe_allow_html=True)

# Load data
data_path = "Cleaned_Nifty_50_Dataset.csv"
df = pd.read_csv(data_path)
df.dropna(inplace=True)

# Features and target
features = ['P/E', 'Div Yield %', 'P/B']
target = 'Price'
X = df[features]
y = df[target]

# Preprocessing
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

# Train and evaluate
results = []
predictions = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
        "R2 Score": r2_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results).sort_values(by="RMSE")

# Sidebar-like dropdown (embedded into screen)
option = st.selectbox(
    "🔍 Choose Visualization",
    [
        "Overall Model Comparison",
        "Actual vs Predicted",
        "Residual Plots",
        "P/E vs Price",
        "Dividend Yield vs Price"
    ]
)

# ========== Visualizations Based on Dropdown ==========
if option == "Overall Model Comparison":
    st.subheader("📊 Overall Model Comparison (MAE, RMSE, R²)")
    fig, ax = plt.subplots(figsize=(10, 5))
    melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Value')
    sns.barplot(data=melted, x="Metric", y="Value", hue="Model", ax=ax)
    for index, row in melted.iterrows():
        ax.text(index % 3 - 0.3, row['Value'] + 30, f"{row['Value']:.2f}", fontsize=7)
    st.pyplot(fig)

elif option == "Actual vs Predicted":
    st.subheader("📌 Actual vs Predicted (All Models)")
    col1, col2 = st.columns(2)
    models_list = list(models.keys())

    for idx, model_name in enumerate(models_list):
        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions[model_name], alpha=0.6)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        ax.set_title(model_name)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        if idx % 2 == 0:
            col1.pyplot(fig)
        else:
            col2.pyplot(fig)

elif option == "Residual Plots":
    st.subheader("📌 Residual Distribution for All Models")
    col1, col2 = st.columns(2)
    for idx, (name, pred) in enumerate(predictions.items()):
        residuals = y_test - pred
        fig, ax = plt.subplots()
        sns.histplot(residuals, bins=30, kde=True, ax=ax)
        ax.set_title(f"Residuals - {name}")
        if idx % 2 == 0:
            col1.pyplot(fig)
        else:
            col2.pyplot(fig)

elif option == "P/E vs Price":
    st.subheader("📌 P/E vs Price")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='P/E', y='Price', color='orange', ax=ax)
    ax.set_title("P/E Ratio vs Price")
    st.pyplot(fig)

elif option == "Dividend Yield vs Price":
    st.subheader("📌 Dividend Yield % vs Price")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Div Yield %', y='Price', color='orange', ax=ax)
    ax.set_title("Dividend Yield % vs Price")
    st.pyplot(fig)

# Hide default Streamlit sidebar (complete removal)
hide_sidebar = """
    <style>
    [data-testid="stSidebar"] { display: none; }
    .css-18e3th9 { padding-top: 2rem; }
    </style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)
