# Linear & Logistic Regression for Profit & Hiring Predictions 📊🔍

## Overview
This **data science project** implements **linear and logistic regression models** to solve two practical business problems: **restaurant profit prediction** based on city population and **candidate hiring prediction** based on interview scores. The implementation uses **Python** with **scikit-learn**, **pandas**, and **matplotlib** to build robust predictive models with clear visualizations.

## Key Features
✔ **Linear Regression Analysis** – Predicts **restaurant profitability** based on **population data** of cities.  
✔ **Logistic Regression Classification** – Determines **hiring decisions** based on **interview performance scores**.  
✔ **Data Visualization** – Creates **scatter plots and regression lines** to illustrate model performance.  
✔ **Multi-class Classification Explanation** – Detailed explanation of **One-vs-Rest** and **One-vs-One** strategies.  

## Project Components
📌 **Python Script** (`main.py`) – Implements linear and logistic regression models with visualizations.  
📌 **Datasets** (`RegressionData.csv`, `LogisticRegressionData.csv`) – Contains training data for both regression tasks.  
📌 **Documentation** – Includes detailed explanations of multi-class classification approaches.  

## Linear Regression: Restaurant Profit Prediction
The first part of the project focuses on:
- Loading and preprocessing population and profit data
- Visualizing data distribution through scatter plots
- Implementing linear regression using least squares optimization
- Predicting restaurant profitability in new locations
- Interpreting model coefficients (intercept and slope)

## Logistic Regression: Candidate Hiring Prediction
The second part of the project demonstrates:
- Binary classification of job candidates based on two interview scores
- Visualization of hiring decisions with color-coded scatter plots
- Training a logistic regression model to predict hiring outcomes
- Evaluating model performance on training data
- Visualizing classification errors

## Multi-class Classification Approaches
The project concludes with detailed explanations of:
- **One-vs-Rest (OvR)** method – Training N binary classifiers for N classes
- **One-vs-One (OvO)** method – Training N*(N-1)/2 binary classifiers for pairwise comparisons
- Strengths and limitations of each approach

## Technologies Used
🔹 **Python** (scikit-learn, pandas, matplotlib)  
🔹 **Linear Regression** (Least squares optimization)  
🔹 **Logistic Regression** (Binary classification)  
🔹 **Data Visualization** (Scatter plots, regression lines)  

## How It Works
1️⃣ The script loads and preprocesses data from CSV files.  
2️⃣ Linear regression model fits the relationship between city population and restaurant profit.  
3️⃣ Logistic regression model learns to classify candidates based on interview scores.  
4️⃣ Visualizations illustrate data patterns and model performance.  
5️⃣ Detailed comparisons of multi-class classification approaches are provided.

## How to Run
📊 **Setup**: `pip install pandas scikit-learn matplotlib numpy`  
🚀 **Execute**: `python main.py`  
📈 **Results**: View console output and visualizations showing linear regression coefficients, profit predictions, and logistic regression classification results
