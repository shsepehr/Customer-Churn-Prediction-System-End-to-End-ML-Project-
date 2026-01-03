import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump

from preprocess import build_preprocessor
from features import add_features

os.makedirs("model", exist_ok=True)

df = pd.read_csv("data/customers.csv")
df = add_features(df)

X = df.drop("churn", axis=1)
y = df["churn"].map({"Yes": 1, "No": 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("preprocess", build_preprocessor()),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

print(classification_report(y_test, pred))

dump(pipeline, "model/churn_model.pkl")
print("✅ Model saved to model/churn_model.pkl")
