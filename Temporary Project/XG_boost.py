import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from a CSV file
data = pd.read_csv('assets/sample.csv')

# Encode categorical variables (one-hot encoding for 'Gender')
data = pd.get_dummies(data, columns=['Gender'], drop_first=True)

X = data[['Age', 'Gender_Male']]  # Use 'Gender_Male' column after one-hot encoding
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("XGBoost Accuracy:", accuracy)
