import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure page
st.set_page_config(page_title="Financial Model Dashboard", layout="wide")

# Title
st.markdown("<h1 style='text-align: center;'>📊 Financial Model Dashboard</h1>", unsafe_allow_html=True)

# Load and prepare data
df = pd.read_csv("Cleaned_Nifty_50_Dataset.csv")
df.dropna(inplace=True)

features = ['P/E', 'Div Yield %', 'P/B']
target = 'Price'
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "KNN": KNeighborsRegressor()
}

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

results_df = pd.DataFrame(results).sort_values(by="RMSE").reset_index(drop=True)

# Hide sidebar
st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    .css-18e3th9 { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

# Dropdown to choose plot
viz_choice = st.selectbox("🔍 Choose Visualization", 
                          ["Overall Comparison", "Actual vs Predicted", 
                           "Residual Plots", "P/E vs Price", "Dividend Yield vs Price"])

# Overall Comparison
if viz_choice == "Overall Comparison":
    st.subheader("📊 Overall Model Performance")
    fig, ax = plt.subplots(figsize=(10, 5))
    melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    sns.barplot(data=melted, x="Metric", y="Value", hue="Model", ax=ax)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.5, f'{height:.2f}', ha='center')
    st.pyplot(fig)

else:
    model_choice = st.selectbox("📌 Select Model", list(models.keys()))

    if viz_choice == "Actual vs Predicted":
        st.subheader(f"📍 Actual vs Predicted - {model_choice}")
        fig1, ax1 = plt.subplots()
        ax1.scatter(y_test, predictions[model_choice], alpha=0.6)
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        ax1.set_xlabel("Actual Price")
        ax1.set_ylabel("Predicted Price")
        ax1.set_title(f"Actual vs Predicted: {model_choice}")
        st.pyplot(fig1)

    elif viz_choice == "Residual Plots":
        st.subheader(f"📍 Residual Distribution - {model_choice}")
        residuals = y_test - predictions[model_choice]
        fig2, ax2 = plt.subplots()
        sns.histplot(residuals, kde=True, bins=30, ax=ax2)
        ax2.set_xlabel("Residual Error")
        ax2.set_title(f"Residuals for {model_choice}")
        st.pyplot(fig2)

    elif viz_choice == "P/E vs Price":
        st.subheader(f"🔎 P/E vs Price for {model_choice}")
        fig3, ax3 = plt.subplots()
        sns.scatterplot(x=df['P/E'], y=df['Price'], ax=ax3)
        ax3.set_xlabel("P/E")
        ax3.set_ylabel("Price")
        ax3.set_title("P/E vs Price")
        st.pyplot(fig3)

    elif viz_choice == "Dividend Yield vs Price":
        st.subheader(f"🔎 Dividend Yield % vs Price for {model_choice}")
        fig4, ax4 = plt.subplots()
        sns.scatterplot(x=df['Div Yield %'], y=df['Price'], ax=ax4)
        ax4.set_xlabel("Div Yield %")
        ax4.set_ylabel("Price")
        ax4.set_title("Dividend Yield % vs Price")
        st.pyplot(fig4)
