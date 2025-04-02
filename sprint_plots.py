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

# Theme Toggle
st.sidebar.title("Settings")
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
        <style>
        body { background-color: #0e1117; color: white; }
        </style>
    """, unsafe_allow_html=True)

st.title("📊 Financial Model Comparison Dashboard")

# Load Data
@st.cache_data

def load_data():
    df = pd.read_csv("Cleaned_Nifty_50_Dataset.csv")
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Preprocess
target = 'Price'
features = ['P/E', 'Div Yield %', 'P/B']
X = df[features]
y = df[target]

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

# Fit and Predict
results = []
predictions = {}
secondary_predictions = {}
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
    # For secondary prediction (e.g., P/E and Dividend)
    for feature in ['P/E', 'Div Yield %']:
        model.fit(X_train_scaled, X_train[feature])
        pred = model.predict(X_test_scaled)
        secondary_predictions[f"{name}_{feature}"] = pred

results_df = pd.DataFrame(results).sort_values(by="RMSE").reset_index(drop=True)

# Main Section Selection
section = st.sidebar.radio("Choose Section", [
    "Overall Comparison", "Model Prediction", "Compare Models"
])

# ---------------- OVERALL COMPARISON -------------------
if section == "Overall Comparison":
    st.subheader("📌 Model Performance Metrics")
    st.dataframe(results_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2 Score": "{:.2f}"}))

    st.subheader("📊 Comparison Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    sns.barplot(data=melted, x="Metric", y="Value", hue="Model", ax=ax)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + 0.05, p.get_height() + 10), fontsize=8)
    st.pyplot(fig)

# ---------------- MODEL PREDICTION ----------------------
elif section == "Model Prediction":
    plot_type = st.selectbox("Select Option", [
        "Actual vs Predicted", "Residual Plots", "Price vs P/E",
        "Price vs Dividend Yield", "Time Series - Price vs P/E",
        "Time Series - Price vs Dividend Yield"
    ])
    model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[model_name]
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    if plot_type == "Actual vs Predicted":
        st.subheader(f"📍 Actual vs Predicted - {model_name}")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        st.pyplot(fig)

    elif plot_type == "Residual Plots":
        st.subheader(f"📍 Residuals - {model_name}")
        residuals = y_test - y_pred
        fig, ax = plt.subplots()
        sns.histplot(residuals, kde=True, color="orange", ax=ax, bins=40, edgecolor="black")
        ax.axvline(x=0, color='black', linestyle='--')
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Residuals Distribution for {model_name}")
        st.pyplot(fig)

    elif plot_type == "Time Series - Price vs P/E":
        st.subheader(f"⏱ Time Series - Price vs P/E - {model_name}")
        dates = df.loc[y_test.index, "Date"]
        pe_pred = secondary_predictions[f"{model_name}_P/E"]
        fig, ax1 = plt.subplots()
        ax1.plot(dates, y_test, label="Actual Price", color='blue')
        ax1.plot(dates, y_pred, label="Predicted Price", color='red')
        ax2 = ax1.twinx()
        ax2.plot(dates, X_test["P/E"], label="Actual P/E", color='green', linestyle="--")
        ax2.plot(dates, pe_pred, label="Predicted P/E", color='orange', linestyle="--")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax2.set_ylabel("P/E")
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        st.pyplot(fig)

    elif plot_type == "Time Series - Price vs Dividend Yield":
        st.subheader(f"⏱ Time Series - Price vs Dividend Yield - {model_name}")
        dates = df.loc[y_test.index, "Date"]
        dy_pred = secondary_predictions[f"{model_name}_Div Yield %"]
        fig, ax1 = plt.subplots()
        ax1.plot(dates, y_test, label="Actual Price", color='blue')
        ax1.plot(dates, y_pred, label="Predicted Price", color='red')
        ax2 = ax1.twinx()
        ax2.plot(dates, X_test["Div Yield %"], label="Actual Dividend", color='purple', linestyle="--")
        ax2.plot(dates, dy_pred, label="Predicted Dividend", color='brown', linestyle="--")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax2.set_ylabel("Dividend Yield")
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        st.pyplot(fig)

# ---------------- MODEL COMPARISON ----------------------
elif section == "Compare Models":
    st.subheader("📍 Compare Models")
    plot_type = st.selectbox("Select Option", [
        "Actual vs Predicted", "Residual Plots", "Price vs P/E",
        "Price vs Dividend Yield", "Time Series - Price vs P/E",
        "Time Series - Price vs Dividend Yield"
    ])
    base_model = "Random Forest"
    compare_model = st.selectbox("Choose Model", [m for m in models if m != base_model])

    col1, col2 = st.columns(2)
    for i, name in enumerate([base_model, compare_model]):
        model = models[name]
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        ax_target = col1 if i == 0 else col2

        with ax_target:
            st.markdown(f"### {name}")

            if plot_type == "Actual vs Predicted":
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, alpha=0.5)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                st.pyplot(fig)

            elif plot_type == "Residual Plots":
                residuals = y_test - y_pred
                fig, ax = plt.subplots()
                sns.histplot(residuals, kde=True, ax=ax, color="brown", bins=40, edgecolor="black")
                ax.axvline(x=0, color='black', linestyle='--')
                ax.set_xlabel("Residuals")
                ax.set_title(f"Residuals Distribution for {name}")
                st.pyplot(fig)

            elif plot_type == "Price vs P/E":
                fig, ax = plt.subplots()
                ax.scatter(y_pred, X_test["P/E"], alpha=0.6)
                ax.plot(np.unique(y_pred), np.poly1d(np.polyfit(y_pred, X_test["P/E"], 1))(np.unique(y_pred)), color="black", linestyle="--")
                st.pyplot(fig)

            elif plot_type == "Price vs Dividend Yield":
                fig, ax = plt.subplots()
                ax.scatter(y_pred, X_test["Div Yield %"], alpha=0.6)
                ax.plot(np.unique(y_pred), np.poly1d(np.polyfit(y_pred, X_test["Div Yield %"], 1))(np.unique(y_pred)), color="black", linestyle="--")
                st.pyplot(fig)

            elif plot_type == "Time Series - Price vs P/E":
                dates = df.loc[y_test.index, "Date"]
                pe_pred = secondary_predictions[f"{name}_P/E"]
                fig, ax1 = plt.subplots()
                ax1.plot(dates, y_test, label="Actual Price", color='blue')
                ax1.plot(dates, y_pred, label="Predicted Price", color='red')
                ax2 = ax1.twinx()
                ax2.plot(dates, X_test["P/E"], label="Actual P/E", color='green', linestyle="--")
                ax2.plot(dates, pe_pred, label="Predicted P/E", color='orange', linestyle="--")
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Price")
                ax2.set_ylabel("P/E")
                fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
                st.pyplot(fig)

            elif plot_type == "Time Series - Price vs Dividend Yield":
                dates = df.loc[y_test.index, "Date"]
                dy_pred = secondary_predictions[f"{name}_Div Yield %"]
                fig, ax1 = plt.subplots()
                ax1.plot(dates, y_test, label="Actual Price", color='blue')
                ax1.plot(dates, y_pred, label="Predicted Price", color='red')
                ax2 = ax1.twinx()
                ax2.plot(dates, X_test["Div Yield %"], label="Actual Dividend", color='purple', linestyle="--")
                ax2.plot(dates, dy_pred, label="Predicted Dividend", color='brown', linestyle="--")
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Price")
                ax2.set_ylabel("Dividend Yield")
                fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
                st.pyplot(fig)