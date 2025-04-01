import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# App layout config
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

# Train-test split and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model setup
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
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # Correct
    r2 = r2_score(y_test, y_pred)
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2 Score": r2})

results_df = pd.DataFrame(results).sort_values(by="RMSE").reset_index(drop=True)

# Section selection
section = st.selectbox("Choose Section", ["Overall Comparison", "Actual vs Predicted", "Residual Plots", "P/E vs Price", "Dividend Yield vs Price"])

# Overall Comparison
if section == "Overall Comparison":
    st.subheader("📈 Model Performance Metrics")

    # Safely format numeric columns only
    styled_df = results_df.copy()
    for col in ['MAE', 'RMSE', 'R2 Score']:
        styled_df[col] = styled_df[col].map("{:.2f}".format)
    st.dataframe(styled_df)

    st.subheader("📊 Performance Comparison Chart")
    melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=melted, x="Metric", y="Value", hue="Model", ax=ax)
    st.pyplot(fig)

# Other sections require model selection
else:
    selected_model = st.selectbox("Select a Model", list(models.keys()))
    y_pred = predictions[selected_model]

    if section == "Actual vs Predicted":
        st.subheader(f"📍 Actual vs Predicted - {selected_model}")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        st.pyplot(fig)

    elif section == "Residual Plots":
        st.subheader(f"📍 Residual Distribution - {selected_model}")
        residuals = y_test - y_pred
        fig, ax = plt.subplots()
        sns.histplot(residuals, kde=True, bins=30, ax=ax)
        ax.set_xlabel("Residual Error")
        st.pyplot(fig)

    elif section == "P/E vs Price":
        st.subheader("📌 P/E vs Price")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='P/E', y='Price', ax=ax)
        st.pyplot(fig)

    elif section == "Dividend Yield vs Price":
        st.subheader("📌 Dividend Yield % vs Price")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='Div Yield %', y='Price', ax=ax)
        st.pyplot(fig)

# Hide default Streamlit sidebar
hide_sidebar_style = """
    <style>
    [data-testid="stSidebar"] {display: none;}
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)
