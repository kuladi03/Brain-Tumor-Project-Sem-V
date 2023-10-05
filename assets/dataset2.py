import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(0)

# Create a balanced dataset with 500 rows for each class
num_samples = 5000

# Generate features
feature_ranges = {
    'Age': (20, 80),
    'Tumor_Size_cm': (0.5, 5.0),
    'Blood_Pressure_mmHg': (90, 140),
    'Cholesterol_mg/dL': (150, 250),
    'Family_History': (0, 1),
    # 'BMI': (18.5, 30.0),
    'Hemoglobin_g/dL': (12.0, 16.0),
    'Blood_Sugar_mg/dL': (80, 120),
    'White_Blood_Cell_Count_K/uL': (4000, 8000),
    'Red_Blood_Cell_Count_million/uL': (4.0, 6.0),
    'Headache': (0, 1),
    'Nausea': (0, 1),
    'Vomiting': (0, 1),
    'Seizures': (0, 1),
    'Memory_Loss': (0, 1),
    'Vision_Problems': (0, 1),
    'Difficulty_Speaking': (0, 1),
    # 'Fatigue': (0, 1),
    # 'Balance_Problems': (0, 1),
    'Coordination_Difficulty': (0, 1),
    'Personality_Changes': (0, 1),
    'Depression': (0, 1),
    'Anxiety': (0, 1),
    'Difficulty_Concentrating': (0, 1),
    # 'Sleep_Problems': (0, 1),
}

data = {feature: [] for feature in feature_ranges.keys()}

# Initialize the 'Tumor_Present' key
data['Tumor_Present'] = []

# Generate the features with real-world values
for _ in range(num_samples):
    for feature, (min_val, max_val) in feature_ranges.items():
        data[feature].append(np.random.uniform(min_val, max_val))
    # Introduce some dependency between age and tumor presence
    age = data['Age'][-1]
    if age < 40 and np.random.rand() < 0.2:  # Lower chance of tumor for younger individuals
        data['Tumor_Present'].append(0)
    elif age >= 40 and np.random.rand() < 0.8:  # Higher chance of tumor for older individuals
        data['Tumor_Present'].append(1)
    else:
        data['Tumor_Present'].append(0)  # Default to no tumor

# Create a DataFrame
df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv('realistic_tumor_dataset.csv', index=False)

# Display the first few rows of the dataset
print(df.head())
