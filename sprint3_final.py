import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Page Config ---
st.set_page_config("Financial Model Comparison Dashboard", layout="wide")

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Nifty_50_Dataset.csv")
    df.dropna(inplace=True)
    return df

df = load_data()

# --- Prepare Data ---
features = ['P/E', 'Div Yield %', 'P/B']
target = 'Price'
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Models ---
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "KNN": KNeighborsRegressor()
}

# --- Evaluation ---
results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2 Score": r2})

results_df = pd.DataFrame(results).sort_values(by="RMSE")
results_df_rounded = results_df.round(2)

# --- App Layout ---
st.title("📊 Financial Model Comparison Dashboard")
tabs = st.tabs(["Overall Comparison", "Model Prediction", "Model Comparison"])

# --- Tab 1: Overall Comparison ---
with tabs[0]:
    st.subheader("📌 Model Performance Metrics")
    st.dataframe(results_df_rounded, use_container_width=True)

    st.subheader("📊 Performance Comparison")
    metrics_df = results_df_rounded.melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=metrics_df, x="Metric", y="Score", hue="Model", ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge")
    st.pyplot(fig)

# --- Tab 2: Model Prediction ---
with tabs[1]:
    st.subheader("📉 Individual Model Visualizations")

    plot_option = st.selectbox("Choose Plot", [
        "Actual vs Predicted", "Residual Plot", "Price vs P/E", "Price vs Dividend Yield"
    ], key="plot_select")

    model_option = st.selectbox("Select Model", list(models.keys()), key="model_select")

    if plot_option == "Actual vs Predicted":
        y_pred = predictions[model_option]
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6, color='teal')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title(f"Actual vs Predicted: {model_option}")
        st.pyplot(fig)

    elif plot_option == "Residual Plot":
        y_pred = predictions[model_option]
        residuals = y_test - y_pred
        fig, ax = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax, color='crimson')
        ax.set_title(f"Residuals for {model_option}")
        ax.set_xlabel("Error")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    elif plot_option == "Price vs P/E":
        fig, ax = plt.subplots()
        sns.scatterplot(x=df['P/E'], y=df['Price'], ax=ax, color='royalblue')
        ax.set_title("P/E Ratio vs Price")
        ax.set_xlabel("P/E")
        ax.set_ylabel("Price")
        st.pyplot(fig)

    elif plot_option == "Price vs Dividend Yield":
        fig, ax = plt.subplots()
        sns.scatterplot(x=df['Div Yield %'], y=df['Price'], ax=ax, color='orange')
        ax.set_title("Dividend Yield vs Price")
        ax.set_xlabel("Dividend Yield (%)")
        ax.set_ylabel("Price")
        st.pyplot(fig)

# --- Tab 3: Model Comparison ---
with tabs[2]:
    st.subheader("🔄 Compare Models")

    comp_option = st.selectbox("Select Comparison Type", [
        "Actual vs Predicted", "Residual Plot", "Price vs P/E", "Price vs Dividend Yield"
    ], key="comparison_type")

    model_to_compare = st.selectbox("Choose Model to Compare with Random Forest", [
        m for m in models.keys() if m != "Random Forest"
    ], key="model_compare")

    col1, col2 = st.columns(2)

    def plot_comp(subplot, model_name, option):
        y_pred = predictions[model_name]
        if option == "Actual vs Predicted":
            subplot.scatter(y_test, y_pred, alpha=0.6, color='green')
            subplot.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
            subplot.set_xlabel("Actual Price")
            subplot.set_ylabel("Predicted Price")
            subplot.set_title(model_name)

        elif option == "Residual Plot":
            residuals = y_test - y_pred
            sns.histplot(residuals, kde=True, ax=subplot, color='purple')
            subplot.set_title(f"Residuals: {model_name}")
            subplot.set_xlabel("Error")

        elif option == "Price vs P/E":
            sns.scatterplot(x=df['P/E'], y=df['Price'], ax=subplot, color='navy')
            subplot.set_title(f"P/E vs Price - {model_name}")
            subplot.set_xlabel("P/E")
            subplot.set_ylabel("Price")

        elif option == "Price vs Dividend Yield":
            sns.scatterplot(x=df['Div Yield %'], y=df['Price'], ax=subplot, color='darkorange')
            subplot.set_title(f"Dividend Yield vs Price - {model_name}")
            subplot.set_xlabel("Dividend Yield (%)")
            subplot.set_ylabel("Price")

    with col1:
        st.write("**Random Forest**")
        fig, ax = plt.subplots()
        plot_comp(ax, "Random Forest", comp_option)
        st.pyplot(fig)

    with col2:
        st.write(f"**{model_to_compare}**")
        fig, ax = plt.subplots()
        plot_comp(ax, model_to_compare, comp_option)
        st.pyplot(fig)
