# Flat-PG-Rent-Prediction

## Overview
This project aims to predict the rent prices for flats or PG accommodations based on various features such as location, type of property, and more.
The predictions are made using machine learning models, and a user-friendly interface is provided through a Streamlit app.


## Datasets
The project uses two datasets:

- Raw Data (data/raw_data.csv): The original dataset containing uncleaned and unprocessed data.

- Cleaned Data (data/cleaned_data.csv): The dataset after preprocessing, including handling missing values and encoding categorical variables.

## Libraries Used

The following libraries were used in this project:

- Pandas: For data manipulation and analysis.

- NumPy: For numerical operations.

- Matplotlib and Seaborn: For data visualization.

- Scikit-learn: For machine learning algorithms and metrics.

- Google Collab Notebook: For interactive data preprocessing and exploration.

- Streamlit: For creating an interactive web application.

## Usage

Enter Property Details: Use the sidebar to input details such as property type, location, and other relevant features.

Predict Rent: Click the "Predict Rent" button to get an estimated rent price.

## Model
The project currently uses the following model:

- Gradient Boosting Regressor: This model was chosen based on its performance metrics during training and testing.

## Results

Model performance metrics are as follows:

- Train R2: 0.9969405038749669
- Test R2: 0.9957105411882591
- Test MSE: 715034.8020481495
- Test RMSE: 845.5973048964557
- Test MAE: 679.807833715253

## Deployment
The Streamlit app is deployed and accessible via this link. This allows users to interact with the model directly in a web browser without needing to run the code locally.

## Contributors
- Aditap Kumar
- Sumit Kumar Sharma