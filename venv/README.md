# Housing Price Regression

## Project Overview

This project demonstrates a machine learning regression pipeline for predicting house prices using structural and location-related features.

The dataset is synthetically generated for educational purposes.

## Objective

Predict housing prices based on:

- Square footage
- Number of bedrooms
- Number of bathrooms
- Property age
- Location tier

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn

## Machine Learning Model

Gradient Boosting Regressor was used for the prediction task.

## Workflow

1. Generate synthetic housing data
2. Create the target price
3. Treat extreme square-footage values
4. Encode the location feature
5. Split the data into training and testing sets
6. Scale the features using RobustScaler
7. Train Gradient Boosting Regressor
8. Generate predictions
9. Evaluate the model

## Evaluation Metrics

- R²
- RMSE
- MAE

## Dataset

The dataset is synthetically generated using NumPy and Pandas.

## Key Learning

This project demonstrates a complete regression workflow, including data preparation, feature preprocessing, model training and regression evaluation.
