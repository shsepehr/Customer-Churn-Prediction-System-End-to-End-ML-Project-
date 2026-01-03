# Customer Churn Prediction System

An end-to-end Machine Learning project to predict customer churn using behavioral, financial, and contract-related data.

## 🚀 Project Overview
Customer churn prediction is a critical task for businesses such as telecom companies, SaaS platforms, and banks.  
This project builds a complete ML pipeline to identify customers who are likely to stop using a service.

## 🧠 Key Features
- Real-world feature engineering
- Handling numerical and categorical data
- Scikit-learn Pipeline & ColumnTransformer
- Model evaluation with classification metrics
- Model persistence using joblib
- Production-ready project structure


## ⚙️ Tech Stack
- Python 3.9+
- pandas, numpy
- scikit-learn
- joblib

## 🧪 Model
- Algorithm: Random Forest Classifier
- Preprocessing:
  - StandardScaler for numerical features
  - OneHotEncoder for categorical features
- Evaluation Metrics:
  - Precision
  - Recall
  - F1-score

## ▶️ How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
2️⃣ Train the model
bash
Copy code
python train.py
3️⃣ Make predictions
bash
Copy code
python predict.py
