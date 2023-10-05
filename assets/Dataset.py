import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(0)

# Create a dictionary with columns and random data
data = {
    'Age': np.random.randint(20, 80, 500),
    'Tumor_Size_cm': np.random.uniform(0.5, 5.0, 500),
    'Blood_Pressure_mmHg': np.random.randint(80, 160, 500),
    'Cholesterol_mg/dL': np.random.randint(120, 300, 500),
    'Family_History': np.random.randint(0, 2, 500),
    'BMI': np.random.uniform(18.5, 35.0, 500),
    'Hemoglobin_g/dL': np.random.uniform(12.0, 17.0, 500),
    'Blood_Sugar_mg/dL': np.random.randint(70, 200, 500),
    'White_Blood_Cell_Count_K/uL': np.random.randint(4000, 11000, 500),
    'Red_Blood_Cell_Count_million/uL': np.random.uniform(4.0, 6.5, 500),
    'Headache': np.random.randint(0, 2, 500),
    'Nausea': np.random.randint(0, 2, 500),
    'Vomiting': np.random.randint(0, 2, 500),
    'Seizures': np.random.randint(0, 2, 500),
    'Memory_Loss': np.random.randint(0, 2, 500),
    'Vision_Problems': np.random.randint(0, 2, 500),
    'Difficulty_Speaking': np.random.randint(0, 2, 500),
    'Fatigue': np.random.randint(0, 2, 500),
    'Balance_Problems': np.random.randint(0, 2, 500),
    'Coordination_Difficulty': np.random.randint(0, 2, 500),
    'Personality_Changes': np.random.randint(0, 2, 500),
    'Depression': np.random.randint(0, 2, 500),
    'Anxiety': np.random.randint(0, 2, 500),
    'Difficulty_Concentrating': np.random.randint(0, 2, 500),
    'Sleep_Problems': np.random.randint(0, 2, 500),
    'Tumor_Present': np.random.randint(0, 2, 500)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv('dummy_dataset.csv', index=False)

# Display the first few rows of the dataset
print(df.head())
