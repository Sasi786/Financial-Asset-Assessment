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

# Load and preprocess data
file_path = "D:/University of Leicester/Dissertation/Assess financial assets market value using data/Cleaned_Nifty_50_Dataset.csv"
df = pd.read_csv(file_path)
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

# Train and evaluate models
results = []
predictions = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)  # Use mean_squared_error without squared=False
    rmse = np.sqrt(mse)  # Manually compute the square root for RMSE
    r2 = r2_score(y_test, y_pred)
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2 Score": r2})

results_df = pd.DataFrame(results).sort_values(by="RMSE")
results_df_rounded = results_df.round(2)

# Streamlit UI
st.set_page_config(page_title="Financial Model Comparison Dashboard")
st.title("📊 Financial Model Comparison Dashboard")

# Tab structure
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

with tabs[1]:
    st.subheader("✅ Individual Model Visualizations")
    plot_type = st.selectbox("Choose Plot", [
        "Actual vs Predicted", "Residual Plots", "Price vs P/E", "Price vs Dividend Yield"
    ], key="ind_plot")
    selected_model = st.selectbox("Select Model", list(models.keys()), key="ind_model")

    if plot_type == "Actual vs Predicted":
        y_pred = predictions[selected_model]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title(f"Actual vs Predicted: {selected_model}")
        st.pyplot(fig)

    elif plot_type == "Residual Plots":
        y_pred = predictions[selected_model]
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(residuals, kde=True, bins=30, ax=ax)
        ax.set_title(f"Residuals for {selected_model}")
        ax.set_xlabel("Error")
        st.pyplot(fig)

    elif plot_type == "Price vs P/E":
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(x=df['P/E'], y=df['Price'], color='orange', ax=ax)
        ax.set_title("P/E vs Price")
        ax.set_xlabel("P/E Ratio")
        ax.set_ylabel("Price")
        st.pyplot(fig)

    elif plot_type == "Price vs Dividend Yield":
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(x=df['Div Yield %'], y=df['Price'], color='orange', ax=ax)
        ax.set_title("Div Yield % vs Price")
        ax.set_xlabel("Dividend Yield (%)")
        ax.set_ylabel("Price")
        st.pyplot(fig)

with tabs[2]:
    st.subheader("🔍 Compare Models")
    comparison_option = st.selectbox("Select Comparison Type", [
        "Actual vs Predicted", "Residual Plots", "Price vs P/E", "Price vs Dividend Yield"
    ], key="compare_option")

    base_model = "Random Forest"
    compare_model = st.selectbox("Choose Model to Compare with Random Forest", [m for m in models.keys() if m != base_model])

    col1, col2 = st.columns(2)

    if comparison_option == "Actual vs Predicted":
        with col1:
            fig, ax = plt.subplots()
            ax.scatter(y_test, predictions[base_model], alpha=0.5)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
            ax.set_title(base_model)
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            ax.scatter(y_test, predictions[compare_model], alpha=0.5)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
            ax.set_title(compare_model)
            st.pyplot(fig)

    elif comparison_option == "Residual Plots":
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(y_test - predictions[base_model], kde=True, bins=30, ax=ax)
            ax.set_title(base_model)
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.histplot(y_test - predictions[compare_model], kde=True, bins=30, ax=ax)
            ax.set_title(compare_model)
            st.pyplot(fig)

    elif comparison_option == "Price vs P/E":
        with col1:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df['P/E'], y=df['Price'], color='orange', ax=ax)
            ax.set_title(base_model)
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df['P/E'], y=df['Price'], color='orange', ax=ax)
            ax.set_title(compare_model)
            st.pyplot(fig)

    elif comparison_option == "Price vs Dividend Yield":
        with col1:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df['Div Yield %'], y=df['Price'], color='orange', ax=ax)
            ax.set_title(base_model)
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df['Div Yield %'], y=df['Price'], color='orange', ax=ax)
            ax.set_title(compare_model)
            st.pyplot(fig)
