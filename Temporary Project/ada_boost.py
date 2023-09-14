import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Load data from a CSV file
data = pd.read_csv('assets/sample.csv')

# Encode categorical variables (one-hot encoding for 'Gender')
data = pd.get_dummies(data, columns=['Gender'], drop_first=True)

X = data[['Age', 'Gender_Male']]  # Use 'Gender_Male' column after one-hot encoding
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an AdaBoostClassifier
adaboost_model = AdaBoostClassifier(n_estimators=50, random_state=42)

# Fit the model to the training data
adaboost_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = adaboost_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("AdaBoost Accuracy:", accuracy)
