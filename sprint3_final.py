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

# --- Streamlit Page Setup ---
st.set_page_config("Financial Model Dashboard", layout="wide")
st.title("📊 Financial Model Comparison Dashboard")

# --- Load Dataset ---
file_path = "Cleaned_Nifty_50_Dataset.csv"  # Ensure this CSV is uploaded on Streamlit Cloud
try:
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
except FileNotFoundError:
    st.error("CSV file not found. Please upload 'Cleaned_Nifty_50_Dataset.csv'.")
    st.stop()

# --- Features and Target ---
features = ['P/E', 'Div Yield %', 'P/B']
target = 'Price'
X = df[features]
y = df[target]

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scaling ---
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

results_df = pd.DataFrame(results)
results_df_rounded = results_df.round(2)

# --- TABS ---
overview, prediction_tab, compare_tab = st.tabs(["Overall Comparison", "Model Prediction", "Model Comparison"])

# ============================== OVERALL TAB ==============================
with overview:
    st.subheader("📌 Model Performance Metrics")
    st.dataframe(results_df_rounded, use_container_width=True)

    st.subheader("📊 Performance Comparison")
    metrics_df = results_df_rounded.melt(id_vars="Model", var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 6))
    chart = sns.barplot(data=metrics_df, x="Metric", y="Score", hue="Model", palette="Set2")
    for p in chart.patches:
        height = p.get_height()
        chart.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                       ha='center', va='bottom', fontsize=8, color='black')
    st.pyplot(plt.gcf())
    plt.clf()

# ============================== MODEL PREDICTION ==============================
with prediction_tab:
    st.subheader("🧠 Individual Model Visualizations")
    comp_plot = st.selectbox("Choose Plot", ["Actual vs Predicted", "Residual Plot", "Price vs P/E", "Price vs Dividend Yield"])
    model_name = st.selectbox("Select Model", list(models.keys()))

    y_pred = predictions[model_name]

    if comp_plot == "Actual vs Predicted":
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7, color='teal')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title(f"Actual vs Predicted - {model_name}")
        st.pyplot(fig)

    elif comp_plot == "Residual Plot":
        residuals = y_test - y_pred
        fig, ax = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax, color='tomato')
        ax.set_title(f"Residuals for {model_name}")
        ax.set_xlabel("Error")
        st.pyplot(fig)

    elif comp_plot == "Price vs P/E":
        fig, ax = plt.subplots()
        sns.scatterplot(x=df['P/E'], y=df['Price'], ax=ax, color='navy')
        ax.set_title("P/E vs Price")
        ax.set_xlabel("P/E Ratio")
        ax.set_ylabel("Price")
        st.pyplot(fig)

    elif comp_plot == "Price vs Dividend Yield":
        fig, ax = plt.subplots()
        sns.scatterplot(x=df['Div Yield %'], y=df['Price'], ax=ax, color='darkgreen')
        ax.set_title("Dividend Yield % vs Price")
        ax.set_xlabel("Dividend Yield (%)")
        ax.set_ylabel("Price")
        st.pyplot(fig)

# ============================== MODEL COMPARISON ==============================
with compare_tab:
    st.subheader("🔄 Compare Models")
    compare_option = st.selectbox("Select Comparison Type", ["Actual vs Predicted", "Residual Plot", "Price vs P/E", "Price vs Dividend Yield"], key="compare_plot")
    model_1 = "Random Forest"
    model_2 = st.selectbox("Choose Model to Compare with Random Forest", [m for m in models if m != model_1], key="compare_model")

    y1, y2 = predictions[model_1], predictions[model_2]

    col1, col2 = st.columns(2)

    if compare_option == "Actual vs Predicted":
        with col1:
            fig, ax = plt.subplots()
            ax.scatter(y_test, y1, alpha=0.7, color='orange')
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
            ax.set_title(f"Actual vs Predicted: {model_1}")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            ax.scatter(y_test, y2, alpha=0.7, color='purple')
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
            ax.set_title(f"Actual vs Predicted: {model_2}")
            st.pyplot(fig)

    elif compare_option == "Residual Plot":
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(y_test - y1, kde=True, ax=ax, color='blue')
            ax.set_title(f"Residuals - {model_1}")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.histplot(y_test - y2, kde=True, ax=ax, color='green')
            ax.set_title(f"Residuals - {model_2}")
            st.pyplot(fig)

    elif compare_option == "Price vs P/E":
        with col1:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df['P/E'], y=df['Price'], ax=ax, color='royalblue')
            ax.set_title(f"P/E vs Price - {model_1}")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df['P/E'], y=df['Price'], ax=ax, color='darkred')
            ax.set_title(f"P/E vs Price - {model_2}")
            st.pyplot(fig)

    elif compare_option == "Price vs Dividend Yield":
        with col1:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df['Div Yield %'], y=df['Price'], ax=ax, color='brown')
            ax.set_title(f"Dividend Yield vs Price - {model_1}")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df['Div Yield %'], y=df['Price'], ax=ax, color='olive')
            ax.set_title(f"Dividend Yield vs Price - {model_2}")
            st.pyplot(fig)
