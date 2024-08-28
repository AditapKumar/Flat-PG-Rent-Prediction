import pandas as pd
import streamlit as st
import joblib

# Load the trained model
model = joblib.load('gradient_boosting_model.pkl')

# Load the original training dataset to get feature names and other info
housing = pd.read_csv('property_rent_data.csv')

# Display information about the dataset
st.write("## Flat/PG Rent Prediction")
st.write("### Sample Dataset")
st.write(housing.head())

# Preprocess categorical variables (one-hot encoding)
X = housing.drop('Rent Price', axis=1)
X = pd.get_dummies(X, drop_first=True)

# Streamlit app for user input and prediction
st.sidebar.title("House Rent Prediction App")

# Input fields for user to enter property details
st.sidebar.header("Enter Property Details")
input_features = {}
for col in X.columns:
    if col.startswith('Type') or col.startswith('Location'):  # Handle categorical columns separately
        original_col = col.split('_')[0]
        if original_col not in input_features:
            input_features[original_col] = st.sidebar.selectbox(original_col, sorted(housing[original_col].unique()))
    else:
        col_min = float(X[col].min()) if not pd.isnull(X[col].min()) else 0.0
        col_max = float(X[col].max()) if not pd.isnull(X[col].max()) else 0.0
        col_mean = float(X[col].mean()) if not pd.isnull(X[col].mean()) else 0.0
        input_features[col] = st.sidebar.number_input(col, min_value=col_min, max_value=col_max, value=col_mean)

# Function to predict rent based on user input
def predict_rent(input_data):
    input_features_df = pd.DataFrame(input_data, index=[0])
    input_features_df = pd.get_dummies(input_features_df, drop_first=True)

    # Align input features with the training data
    missing_cols = set(X.columns) - set(input_features_df.columns)
    for col in missing_cols:
        input_features_df[col] = 0
    input_features_df = input_features_df[X.columns]

    # Predict rent price
    predicted_rent = model.predict(input_features_df)
    return predicted_rent[0]

# Predict button
if st.sidebar.button("Predict Rent"):
    input_data = pd.DataFrame([input_features])
    predicted_price = predict_rent(input_data)
    st.sidebar.success(f"Predicted Rent Price: Rs.{predicted_price:.2f}")