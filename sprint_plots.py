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

# Streamlit UI for file upload
st.set_page_config(page_title="Financial Model Comparison Dashboard")
st.title("📊 Financial Model Comparison Dashboard")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.dropna(inplace=True)

    # Prepare features and target
    features = ['P/E', 'Div Yield %', 'P/B']
    target = 'Price'
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model definitions
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "KNN": KNeighborsRegressor()
    }

    # Train and evaluate models
    results = []
    predictions = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        predictions[name] = y_pred
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)  # Compute RMSE manually
        r2 = r2_score(y_test, y_pred)
        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2 Score": r2})

    results_df = pd.DataFrame(results).sort_values(by="RMSE")
    results_df_rounded = results_df.round(2)

    # Display results and metrics
    tabs = st.tabs(["Overall Comparison", "Model Prediction", "Model Comparison"])
    
    with tabs[0]:
        st.subheader("📌 Model Performance Metrics")
        st.dataframe(results_df_rounded, use_container_width=True)
        
        st.subheader("📊 Performance Comparison")
        metrics_df = results_df_rounded.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=metrics_df, x="Metric", y="Score", hue="Model", ax=ax)
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=8, color='black', xytext=(0, 5),
                        textcoords='offset points')
        st.pyplot(fig)
else:
    st.warning("Please upload a CSV file.")
