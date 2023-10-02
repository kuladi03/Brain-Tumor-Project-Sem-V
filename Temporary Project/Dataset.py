import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(0)

# Define the number of entries in the dataset
num_entries = 1000

# Generate random data for features with actual names
data = {
    'Age': np.random.randint(20, 80, num_entries),  # Age between 20 and 80 years
    'Tumor_Size_cm': np.random.uniform(0.5, 5.0, num_entries),  # Tumor size in cm
    'Blood_Pressure_mmHg': np.random.randint(80, 160, num_entries),  # Blood pressure in mmHg
    'Cholesterol_mg/dL': np.random.randint(120, 300, num_entries),  # Cholesterol level in mg/dL
    'Family_History': np.random.randint(0, 2, num_entries),  # 0 for no family history, 1 for family history
    'BMI': np.random.uniform(18.5, 35.0, num_entries),  # BMI (Body Mass Index)
    'Hemoglobin_g/dL': np.random.uniform(12.0, 17.0, num_entries),  # Hemoglobin level in g/dL
    'Blood_Sugar_mg/dL': np.random.randint(70, 200, num_entries),  # Blood sugar level in mg/dL
    'White_Blood_Cell_Count_K/uL': np.random.randint(4000, 11000, num_entries),  # WBC count in K/uL
    'Red_Blood_Cell_Count_million/uL': np.random.uniform(4.0, 6.5, num_entries),  # RBC count in million/uL
}

# Generate random labels for binary classification (0 for no tumor, 1 for tumor)
labels = np.random.randint(2, size=num_entries)

# Create a DataFrame
df = pd.DataFrame(data)
df['Tumor_Present'] = labels  # Binary label: 0 for no tumor, 1 for tumor

# Display the first few rows of the dataset
print(df.head())

# Save the dataset to a CSV file if needed
df.to_csv('tumor_detection_dataset.csv', index=False)
