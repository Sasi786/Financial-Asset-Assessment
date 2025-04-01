import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Streamlit Page Config ---
st.set_page_config(page_title="Financial Model Dashboard", layout="wide")
st.title("📊 Financial Model Comparison Dashboard")

# --- Load Dataset ---
data_path = "Cleaned_Nifty_50_Dataset.csv"
df = pd.read_csv(data_path)
df.dropna(inplace=True)

features = ['P/E', 'Div Yield %', 'P/B']
target = 'Price'

X = df[features]
y = df[target]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Define Models ---
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "KNN": KNeighborsRegressor()
}

# --- Evaluate ---
results = []
predictions = {}

for name, model in models.items():
    try:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        predictions[name] = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5  # ✅ FIXED HERE
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2 Score": round(r2, 2)
        })
    except Exception as e:
        st.warning(f"⚠️ Model {name} failed: {e}")

results_df = pd.DataFrame(results)
if not results_df.empty:
    results_df = results_df.sort_values(by="RMSE").reset_index(drop=True)

# --- Streamlit UI ---
section = st.selectbox("Choose Section", ["Overall Comparison", "Actual vs Predicted", "Residual Plots", "P/E vs Price", "Dividend Yield vs Price"])

if section == "Overall Comparison":
    st.subheader("📌 Model Performance Metrics")
    st.dataframe(results_df)

    # Comparison Plot
    if not results_df.empty:
        st.subheader("📈 Performance Comparison")
        melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=melted, x="Metric", y="Value", hue="Model", ax=ax)
        st.pyplot(fig)

else:
    if not predictions:
        st.error("No valid model predictions to display.")
    else:
        model_choice = st.selectbox("Select Model", list(predictions.keys()))

        if section == "Actual vs Predicted":
            st.subheader(f"📍 Actual vs Predicted - {model_choice}")
            fig, ax = plt.subplots()
            ax.scatter(y_test, predictions[model_choice], alpha=0.6)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
            ax.set_xlabel("Actual Price")
            ax.set_ylabel("Predicted Price")
            st.pyplot(fig)

        elif section == "Residual Plots":
            st.subheader(f"📍 Residual Distribution - {model_choice}")
            residuals = y_test - predictions[model_choice]
            fig, ax = plt.subplots()
            sns.histplot(residuals, kde=True, ax=ax)
            ax.set_title(f"Residuals: {model_choice}")
            st.pyplot(fig)

        elif section == "P/E vs Price":
            st.subheader(f"📍 P/E Ratio vs Price - {model_choice}")
            fig, ax = plt.subplots()
            ax.scatter(df['P/E'], df['Price'], alpha=0.6)
            ax.set_xlabel("P/E Ratio")
            ax.set_ylabel("Price")
            st.pyplot(fig)

        elif section == "Dividend Yield vs Price":
            st.subheader(f"📍 Dividend Yield % vs Price - {model_choice}")
            fig, ax = plt.subplots()
            ax.scatter(df['Div Yield %'], df['Price'], alpha=0.6)
            ax.set_xlabel("Dividend Yield %")
            ax.set_ylabel("Price")
            st.pyplot(fig)

# --- Hide Sidebar ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    </style>
""", unsafe_allow_html=True)
