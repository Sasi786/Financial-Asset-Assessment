import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_path = "D:/University of Leicester/Dissertation/Assess financial assets market value using data/Cleaned_Nifty_50_Dataset.csv"
df = pd.read_csv(file_path)
df.dropna(inplace=True)

# Features and target
features = ['P/E', 'Div Yield %', 'P/B']
target = 'Price'
X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
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

# Color mapping per model
model_colors = {
    "Linear Regression": "royalblue",
    "Random Forest": "darkgreen",
    "Gradient Boosting": "purple",
    "Decision Tree": "orangered",
    "KNN": "goldenrod"
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
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2 Score": r2})

results_df = pd.DataFrame(results)

# Remove formatting error by applying numeric format only to numeric columns
styled_results_df = results_df.copy()
for col in styled_results_df.columns:
    if styled_results_df[col].dtype in ['float64', 'int64']:
        styled_results_df[col] = styled_results_df[col].map('{:.2f}'.format)

# Dashboard
st.set_page_config(page_title="Financial Model Dashboard", layout="wide")
st.title("\U0001F4CA Financial Model Comparison Dashboard")

overview, prediction, comparison = st.tabs(["Overall Comparison", "Model Prediction", "Model Comparison"])

# --- Overall Tab ---
with overview:
    st.subheader("\U0001F4CC Model Performance Metrics")
    st.dataframe(styled_results_df, use_container_width=True)

    st.subheader("\U0001F4CA Performance Comparison")
    metrics_df = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=metrics_df, x="Metric", y="Score", hue="Model", ax=ax)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 5),
                    textcoords='offset points')
    st.pyplot(fig)

# --- Model Prediction Tab ---
with prediction:
    st.subheader("\U0001F4CB Individual Model Visualizations")
    plot_type = st.selectbox("Choose Plot", ["Actual vs Predicted", "Residual Plot", "Price vs P/E", "Price vs Dividend Yield"])
    model_name = st.selectbox("Select Model", list(models.keys()))

    if plot_type == "Actual vs Predicted":
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, predictions[model_name], alpha=0.6, color=model_colors.get(model_name, "gray"))
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title(f"Actual vs Predicted: {model_name}")
        st.pyplot(fig)

    elif plot_type == "Residual Plot":
        fig, ax = plt.subplots(figsize=(6, 4))
        residuals = y_test - predictions[model_name]
        sns.histplot(residuals, kde=True, ax=ax, bins=30, color=model_colors.get(model_name, "gray"))
        ax.set_title(f"Residuals for {model_name}")
        ax.set_xlabel("Error")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    elif plot_type == "Price vs P/E":
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(x=df['P/E'], y=df['Price'], color="teal", ax=ax)
        ax.set_title("P/E vs Price")
        ax.set_xlabel("P/E Ratio")
        ax.set_ylabel("Price")
        st.pyplot(fig)

    elif plot_type == "Price vs Dividend Yield":
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(x=df['Div Yield %'], y=df['Price'], color="darkorange", ax=ax)
        ax.set_title("Div Yield % vs Price")
        ax.set_xlabel("Dividend Yield (%)")
        ax.set_ylabel("Price")
        st.pyplot(fig)

# --- Model Comparison Tab ---
with comparison:
    st.subheader("\U0001F4E2 Compare Models")
    compare_plot = st.selectbox("Select Comparison Type", ["Actual vs Predicted", "Residual Plot"])
    model_to_compare = st.selectbox("Choose Model to Compare with Random Forest", [m for m in models if m != "Random Forest"])

    col1, col2 = st.columns(2)

    if compare_plot == "Actual vs Predicted":
        with col1:
            st.markdown("**Random Forest**")
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            ax1.scatter(y_test, predictions["Random Forest"], alpha=0.6, color=model_colors.get("Random Forest"))
            ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
            ax1.set_xlabel("Actual Price")
            ax1.set_ylabel("Predicted Price")
            st.pyplot(fig1)

        with col2:
            st.markdown(f"**{model_to_compare}**")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.scatter(y_test, predictions[model_to_compare], alpha=0.6, color=model_colors.get(model_to_compare))
            ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
            ax2.set_xlabel("Actual Price")
            ax2.set_ylabel("Predicted Price")
            st.pyplot(fig2)

    elif compare_plot == "Residual Plot":
        with col1:
            st.markdown("**Random Forest**")
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            residuals_rf = y_test - predictions["Random Forest"]
            sns.histplot(residuals_rf, kde=True, ax=ax1, bins=30, color=model_colors.get("Random Forest"))
            ax1.set_title("Residuals: Random Forest")
            st.pyplot(fig1)

        with col2:
            st.markdown(f"**{model_to_compare}**")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            residuals_cm = y_test - predictions[model_to_compare]
            sns.histplot(residuals_cm, kde=True, ax=ax2, bins=30, color=model_colors.get(model_to_compare))
            ax2.set_title(f"Residuals: {model_to_compare}")
            st.pyplot(fig2)
