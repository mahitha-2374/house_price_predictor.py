import streamlit as st
     import pandas as pd
     import numpy as np
     from sklearn.datasets import fetch_california_housing
     from sklearn.model_selection import train_test_split
     from sklearn.preprocessing import StandardScaler
     from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
     from sklearn.metrics import mean_squared_error, r2_score
     import matplotlib.pyplot as plt
     import seaborn as sns
     import shap
     import joblib

     # Set page configuration
     st.set_page_config(page_title="California House Price Predictor", page_icon="🏠", layout="wide")

     # Cache the model and data loading
     @st.cache_resource
     def load_data_and_model(model_type="rf", n_estimators=100, max_depth=None):
         # Load California Housing dataset
         data = fetch_california_housing()
         df = pd.DataFrame(data.data, columns=data.feature_names)
         df['MedHouseVal'] = data.target * 100000  # Scale to dollars
         
         # Feature engineering
         df['RoomsPerHouse'] = df['AveRooms'] / df['HouseAge']
         df['BedroomRatio'] = df['AveBedrms'] / df['AveRooms']
         
         X = df.drop('MedHouseVal', axis=1)
         y = df['MedHouseVal']
         
         # Split data
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
         
         # Scale features
         scaler = StandardScaler()
         X_train_scaled = scaler.fit_transform(X_train)
         X_test_scaled = scaler.transform(X_test)
         
         # Train model
         if model_type == "rf":
             model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
         else:
             model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
         
         model.fit(X_train_scaled, y_train)
         
         # Calculate metrics
         y_pred = model.predict(X_test_scaled)
         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
         r2 = r2_score(y_test, y_pred)
         
         return model, scaler, X_train.columns, rmse, r2, X_train, y_train

     # Initialize SHAP explainer with unhashable parameter fix
     @st.cache_resource
     def init_shap_explainer(_model, X_train_scaled):
         explainer = shap.TreeExplainer(_model)
         return explainer

     # App title and description
     st.title("🏠 California House Price Predictor")
     st.write("""
         Predict house prices in California using advanced machine learning models.
         Adjust input features and model parameters to get accurate predictions and detailed insights.
     """)

     # Sidebar for model selection and parameters
     st.sidebar.header("Model Configuration")
     model_type = st.sidebar.selectbox("Model Type", ["Random Forest", "Gradient Boosting"], index=0)
     n_estimators = st.sidebar.slider("Number of Trees", 50, 200, 100, step=10)
     max_depth = st.sidebar.slider("Max Tree Depth", 3, 15, 10, step=1)

     # Load model and data
     model_key = f"{model_type}_{n_estimators}_{max_depth}"
     model, scaler, feature_names, rmse, r2, X_train, y_train = load_data_and_model(
         model_type="rf" if model_type == "Random Forest" else "gb",
         n_estimators=n_estimators,
         max_depth=max_depth
     )
     explainer = init_shap_explainer(model, scaler.transform(X_train))

     # Input form for house features
     st.header("Input House Features")
     col1, col2 = st.columns(2)

     with col1:
         med_inc = st.slider("Median Income (in tens of thousands)", 0.5, 15.0, 3.0, 0.1)
         house_age = st.slider("House Age (years)", 1, 52, 25, 1)
         ave_rooms = st.slider("Average Rooms", 1.0, 10.0, 5.0, 0.1)
         ave_bedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0, 0.1)

     with col2:
         population = st.slider("Population", 100, 5000, 1000, 100)
         ave_occup = st.slider("Average Occupancy", 1.0, 10.0, 3.0, 0.1)
         latitude = st.slider("Latitude", 32.5, 42.0, 37.0, 0.1)
         longitude = st.slider("Longitude", -124.5, -114.0, -120.0, 0.1)

     # Feature engineering for input
     input_df = pd.DataFrame({
         'MedInc': [med_inc], 'HouseAge': [house_age], 'AveRooms': [ave_rooms],
         'AveBedrms': [ave_bedrms], 'Population': [population], 'AveOccup': [ave_occup],
         'Latitude': [latitude], 'Longitude': [longitude]
     })
     input_df['RoomsPerHouse'] = input_df['AveRooms'] / input_df['HouseAge']
     input_df['BedroomRatio'] = input_df['AveBedrms'] / input_df['AveRooms']

     # Make prediction
     if st.button("Predict House Price"):
         input_scaled = scaler.transform(input_df)
         prediction = model.predict(input_scaled)[0]
         
         # Display prediction
         st.header("Prediction Results")
         st.success(f"Predicted House Price: **${prediction:,.2f}**")
         
         # Display model metrics
         st.subheader("Model Performance")
         st.write(f"Root Mean Squared Error: ${rmse:,.2f}")
         st.write(f"R² Score: {r2:.3f}")
         
         # SHAP feature importance
         st.subheader("Feature Importance (SHAP)")
         shap_values = explainer.shap_values(input_scaled)
         fig, ax = plt.subplots()
         shap.summary_plot(shap_values, input_df, feature_names=feature_names, plot_type="bar", show=False)
         st.pyplot(fig)
         
         # Prediction distribution
         st.subheader("Prediction Distribution")
         y_pred_all = model.predict(scaler.transform(X_train))
         fig, ax = plt.subplots()
         sns.histplot(y_pred_all, kde=True, ax=ax)
         ax.axvline(prediction, color='red', linestyle='--', label='Current Prediction')
         ax.set_xlabel("House Price ($)")
         ax.legend()
         st.pyplot(fig)

     # Model information
     with st.expander("About the Model"):
         st.write("""
             - **Dataset**: California Housing Dataset
             - **Features**: Median Income, House Age, Average Rooms, Average Bedrooms, Population, Average Occupancy, Latitude, Longitude, plus engineered features
             - **Models**: Random Forest or Gradient Boosting Regressor
             - **Preprocessing**: StandardScaler for feature scaling
             - **Interpretability**: SHAP values for feature importance
             - **Metrics**: RMSE and R² for model evaluation
         """)
