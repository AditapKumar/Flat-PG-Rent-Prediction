import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load the dataset
housing = pd.read_csv('property_rent_data.csv')

# Check for missing values
missing_values = housing.isnull().sum()
if missing_values.any():
    print("Dataset contains missing values. Please preprocess accordingly.")

# Separate features and target variable
X = housing.drop('Rent Price', axis=1)  # Features
y = housing['Rent Price']  # Target variable

# Preprocess categorical variables (one-hot encoding)
X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical variables

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
# model = LinearRegression()

# Random Forest Regressor model
# model = RandomForestRegressor()

# Gradient Boosting model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Evaluate the model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"Train R2: {train_r2}")
print(f"Test R2: {test_r2}")
print(f"Test MSE: {test_mse}")
print(f"Test RMSE: {test_rmse}")
print(f"Test MAE: {test_mae}")


# Save the trained model for later use
joblib.dump(model, 'gradient_boosting_model.pkl')