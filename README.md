🛍️ Customer Product Purchase Prediction (ProdTaken) — ML + Streamlit

This project predicts whether a customer will purchase a product (ProdTaken) based on demographic, behavioral and marketing interaction features.

📌 Project Workflow

Data Preprocessing

Handling missing values (Mean, Median & Mode imputation)

Feature selection and splitting (X and Y)

Encoding categorical features using One-Hot Encoding

Scaling numerical features using StandardScaler

Model Training

Trained and compared multiple ML models:

Logistic Regression

Decision Tree

Random Forest

AdaBoost

Gradient Boosting

XGBoost

Random Forest achieved best performance

Model Saving

Exported using joblib

RF_Classifier_model.joblib
preprocessor_file.joblib


Deployment

Built Web Interface using Streamlit

User inputs customer details → model predicts whether the customer will purchase the product

🚀 How to Run Locally

pip install -r requirements.txt

streamlit run app.py
