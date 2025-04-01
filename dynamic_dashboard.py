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

# Streamlit config
st.set_page_config(page_title="Financial Model Dashboard", layout="wide")
st.title("📊 Financial Model Comparison Dashboard")

# Load dataset
data_path = "Cleaned_Nifty_50_Dataset.csv"
df = pd.read_csv(data_path)
df.dropna(inplace=True)

# Features and target
features = ['P/E', 'Div Yield %', 'P/B']
target = 'Price'
X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "KNN": KNeighborsRegressor()
}

# Store predictions and fitted models
results = []
predictions = {}
fitted_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    fitted_models[name] = model
    predictions[name] = y_pred
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    results.append({"Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2 Score": round(r2, 2)})

results_df = pd.DataFrame(results).sort_values(by="RMSE").reset_index(drop=True)

# Dashboard dropdown
view_option = st.selectbox("Choose Section", [
    "Overall Comparison",
    "Actual vs Predicted",
    "Residual Plots",
    "P/E vs Price",
    "Dividend Yield vs Price"
])

# Overall Comparison
if view_option == "Overall Comparison":
    st.subheader("✅ Model Performance Metrics")
    st.dataframe(results_df)

    st.subheader("📊 Performance Comparison Chart")
    fig, ax = plt.subplots(figsize=(10, 6))
    melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    sns.barplot(data=melted, x="Metric", y="Value", hue="Model", ax=ax)
    for index, row in melted.iterrows():
        ax.text(x=index % 3 - 0.3, y=row['Value'] + 20, s=f"{row['Value']:.2f}", fontsize=8)
    st.pyplot(fig)

# Other sections require a model to be selected
else:
    selected_model = st.selectbox("Select a Model", list(models.keys()))
    model = fitted_models[selected_model]
    y_pred = predictions[selected_model]

    if view_option == "Actual vs Predicted":
        st.subheader(f"📌 Actual vs Predicted: {selected_model}")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title(f"Actual vs Predicted: {selected_model}")
        st.pyplot(fig)

    elif view_option == "Residual Plots":
        st.subheader(f"📍 Residual Plot - {selected_model}")
        residuals = y_test - y_pred
        fig, ax = plt.subplots()
        sns.histplot(residuals, kde=True, bins=30, ax=ax)
        ax.set_title(f"Residuals for {selected_model}")
        ax.set_xlabel("Error")
        st.pyplot(fig)

    elif view_option == "P/E vs Price":
        st.subheader("📈 P/E vs Price")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df['P/E'], y=df['Price'], ax=ax)
        ax.set_title("P/E vs Price")
        ax.set_xlabel("P/E")
        ax.set_ylabel("Price")
        st.pyplot(fig)

    elif view_option == "Dividend Yield vs Price":
        st.subheader("📈 Dividend Yield % vs Price")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df['Div Yield %'], y=df['Price'], ax=ax)
        ax.set_title("Dividend Yield % vs Price")
        ax.set_xlabel("Dividend Yield %")
        ax.set_ylabel("Price")
        st.pyplot(fig)

# Hide Streamlit's floating bar
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="stToolbar"] { display: none; }
    </style>
""", unsafe_allow_html=True)
