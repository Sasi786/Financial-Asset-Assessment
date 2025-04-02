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

# --- Load Dataset from GitHub ---
url = "https://raw.githubusercontent.com/Sasi786/Financial-Asset/main/Cleaned_Nifty_50_Dataset.csv"
df = pd.read_csv(url)
df.dropna(inplace=True)

features = ['P/E', 'Div Yield %', 'P/B']
target = 'Price'
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Define Models ---
from sklearn.base import RegressorMixin

model_defs = {
    "Linear Regression": LinearRegression,
    "Random Forest": lambda: RandomForestRegressor(random_state=42),
    "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=42),
    "Decision Tree": lambda: DecisionTreeRegressor(random_state=42),
    "KNN": KNeighborsRegressor
}

results = []
predictions = {}

for model_name, model_constructor in model_defs.items():
    try:
        model: RegressorMixin = model_constructor()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        predictions[model_name] = y_pred
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        results.append({"Model": model_name, "MAE": mae, "RMSE": rmse, "R2 Score": r2})
    except Exception as e:
        st.error(f"Model {model_name} failed with error: {e}")

results_df = pd.DataFrame(results).sort_values(by="RMSE")
results_df_rounded = results_df.round(2)

# --- Streamlit UI ---
st.set_page_config(page_title="Financial Model Comparison Dashboard")
st.title("📊 Financial Model Comparison Dashboard")

tabs = st.tabs(["Overall Comparison", "Model Prediction", "Model Comparison"])

# --- Tab 1: Overall Comparison ---
with tabs[0]:
    st.subheader("📌 Model Performance Metrics")
    st.dataframe(results_df_rounded, use_container_width=True)

    st.subheader("📊 Performance Comparison")
    if not results_df_rounded.empty:
        metrics_df = results_df_rounded.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=metrics_df, x="Metric", y="Score", hue="Model", ax=ax, palette="Set2")
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=8, color='black', xytext=(0, 5),
                        textcoords='offset points')
        st.pyplot(fig)
    else:
        st.warning("No model results to display.")

# --- Tab 2: Individual Visualizations ---
with tabs[1]:
    st.subheader("✅ Individual Model Visualizations")
    plot_type = st.selectbox("Choose Plot", [
        "Actual vs Predicted", "Residual Plots", "Price vs P/E", "Price vs Dividend Yield"
    ])
    selected_model = st.selectbox("Select Model", list(predictions.keys()))

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
        y_pred = predictions[selected_model]
        residuals = y_test - y_pred
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
        ax.set_title("Div Yield % vs Price")
        ax.set_xlabel("Dividend Yield (%)")
        ax.set_ylabel("Price")
        st.pyplot(fig)

# --- Tab 3: Compare Models ---
with tabs[2]:
    st.subheader("🔍 Compare Models")
    if not predictions:
        st.warning("Model predictions not available due to earlier error.")
    else:
        comparison_option = st.selectbox("Select Comparison Type", [
            "Actual vs Predicted", "Residual Plots", "Price vs P/E", "Price vs Dividend Yield"
        ])
        base_model = "Random Forest"
        compare_model = st.selectbox("Choose Model to Compare with Random Forest", [m for m in predictions.keys() if m != base_model])

        col1, col2 = st.columns(2)

        if comparison_option == "Actual vs Predicted":
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

        elif comparison_option == "Residual Plots":
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

        elif comparison_option == "Price vs P/E":
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

        elif comparison_option == "Price vs Dividend Yield":
            with col1:
                fig, ax = plt.subplots()
                sns.scatterplot(x=df['Div Yield %'], y=df['Price'], color='goldenrod', ax=ax)
                ax.set_title(base_model)
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots()
                sns.scatterplot(x=df['Div Yield %'], y=df['Price'], color='deepskyblue', ax=ax)
                ax.set_title(compare_model)
                st.pyplot(fig)
