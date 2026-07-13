# Bike Demand Prediction in Sochi: Non-Linear Models vs. Weather

<img width="3000" height="1688" alt="image" src="https://github.com/user-attachments/assets/e4cca9f9-668d-42d9-b560-5722a25eec44" />

A machine learning project aimed at improving the hourly bike rental demand forecasting system for "BikeSochi" by transitioning from a basic linear model to non-linear algorithms capable of capturing complex weather dependencies.

## Project Description
"BikeSochi," an urban bike rental operator, faced issues with their existing Linear Regression model. The baseline model failed to understand non-linear weather impacts—for example, it could not distinguish between +28 °C under a scorching sun and +28 °C after rain, predicting high demand for both simply because the temperature was high. 

This project explores and implements non-linear machine learning models (k-Nearest Neighbors and Decision Trees) to better capture the intricate relationships between weather conditions (temperature, humidity, solar radiation, rainfall) and rental demand.

## Goals and Objectives
- Perform Exploratory Data Analysis (EDA) to understand the distribution of demand and its correlation with weather and time features.
- Evaluate the baseline Linear Regression model provided by the company.
- Train non-linear models: k-Nearest Neighbors (kNN) Regressor and Decision Tree Regressor.
- Optimize hyperparameters for both models using the Optuna framework to minimize the Root Mean Squared Error (RMSE).
- Compare models using cross-validation and select the best one for production deployment.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn (LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, Pipeline, ColumnTransformer)
- Optuna (for hyperparameter tuning)
- Matplotlib, Seaborn

## Project Structure
- `bike_demand_prediction.ipynb` — The main Jupyter Notebook containing data analysis, model training, hyperparameter optimization, and business interpretation.
- `best_knn_pipeline.joblib` — The saved optimal machine learning pipeline ready for inference (generated after running the notebook).

## Results
The project successfully demonstrated that non-linear models significantly outperform the baseline linear regression. The optimized **kNN model** was selected as the best solution, achieving an RMSE of ~312 on the test set (compared to ~442 for the baseline). This means the typical prediction error was reduced from 315 to 211 bikes per hour. The final model is saved as a pipeline and is ready for integration into BikeSochi's production environment.
