import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Load the tumor dataset
data = pd.read_csv("assets/realistic_tumor_dataset.csv")

# Select the features we want to use for training
selected_features = [
    'Age',
    'Tumor_Size_cm',
    'Blood_Pressure_mmHg',
    'Cholesterol_mg/dL',
    'Family_History',
    # 'BMI',
    'Hemoglobin_g/dL',
    'Blood_Sugar_mg/dL',
    'White_Blood_Cell_Count_K/uL',
    'Red_Blood_Cell_Count_million/uL'
]

# Split the data into training and testing sets
X = data[selected_features]
y = data['Tumor_Present']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data so that all features have the same scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a dictionary of classifiers to try
classifiers = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(),
    'SVM': SVC(probability=True)
}

# Tune the hyperparameters of each classifier using grid search
best_models = {}
best_accuracies = {}

# Store accuracies for all classifiers
all_accuracies = {}

for name, classifier in classifiers.items():
    param_grids = {}
    if name == 'Decision Tree':
        param_grids = {'criterion': ['gini', 'entropy']}
    elif name == 'Random Forest':
        param_grids = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    elif name == 'Logistic Regression':
        param_grids = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    elif name == 'K-Nearest Neighbors':
        param_grids = {'n_neighbors': [3, 5, 7, 9]}
    elif name == 'AdaBoost':
        param_grids = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
    elif name == 'Gradient Boosting':
        param_grids = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
    elif name == 'XGBoost':
        param_grids = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
    elif name == 'SVM':
        param_grids = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

    grid_search = GridSearchCV(classifier, param_grids, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_models[name] = grid_search.best_estimator_
    y_pred = best_models[name].predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    best_accuracies[name] = accuracy
    all_accuracies[name] = accuracy

# Find the best classifier based on accuracy
best_classifier = max(best_accuracies, key=best_accuracies.get)

print(f"The best classifier is {best_classifier} with accuracy {best_accuracies[best_classifier] * 100:.2f}%")

# Print accuracies for all classifiers
print("\nAccuracies for all classifiers:")
for name, accuracy in all_accuracies.items():
    print(f"{name}: {accuracy * 100:.2f}%")
