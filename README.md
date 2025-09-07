# Predict Bank Marketing Campaign
This repository contains a complete Machine Learning pipeline to predict the success of a bank marketing campaign (whether a client subscribes to a term deposit). It compares several classifiers: Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost, LightGBM and includes preprocessing, hyperparameter tuning, evaluation, and explainability (SHAP).

## Goal
Objective: predict the target y (binary: yes/no) from customer attributes and campaign/economic context.

Key goals:
- Compare performance for different models.
- Tune models via RandomizedSearchCV and Hyperopt.
- Use AUC-ROC as the primary metric; inspect confusion matrix and classification report.
- Explain predictions with feature importance and SHAP.

## Dataset
Bank Marketing (“bank-additional”) data (Portuguese bank phone marketing).
Typical columns:
- Demographics: age, job, marital, education, default, housing, loan
- Campaign history: contact, month, day_of_week, campaign, pdays, previous, poutcome
- Economics: emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed
- Target: y (binary: 'yes' = 1, 'no' = 0)
- Note: The notebook drops duration (leakage if prediction is made before the call ends).

## Project Overview
The project includes full-cycle machine learning development:
- Preprocessing pipelines
- Model training and evaluation
- Hyperparameter tuning
- Explainability

## Preprocessing
- Column selection: all features except duration; target is y.
- For LR/KNN -> numeric MinMaxScaler + OneHotEncoder for categoricals.
- For Tree/BBoosting -> OneHotEncoder for categoricals only.

## Model Training and Evaluation
The models are trained using cross-validation. Evaluation includes:
- AUC Score: Main metric for model performance.
- Confusion Matrix: To analyze misclassifications.

## Results
| Model Name             | AUROC on Train | AUROC on Validation |
| ---------------------- | -------------- | ------------------- |
| LogisticRegression     |0.79            |0.80                 |
| KNeighborsClassifier   |0.89            |0.75                 |
| DecisionTreeClassifier |0.72            |0.71                 |
| RandomForestClassifier |0.76            |0.76                 |
| XGBClassifier          |0.78            |0.77                 |
| LGBMClassifier         |0.77            |0.77                 |
