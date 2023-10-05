import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(0)

# Create a balanced dataset with 500 rows for each class
num_samples = 3000

# Generate features
age = np.random.randint(20, 80, num_samples)
tumor_size = np.random.uniform(0.5, 5.0, num_samples)
blood_pressure = np.random.randint(90, 140, num_samples)
cholesterol = np.random.randint(150, 250, num_samples)
family_history = np.random.randint(0, 2, num_samples)
bmi = np.random.uniform(18.5, 30.0, num_samples)
hemoglobin = np.random.uniform(12.0, 16.0, num_samples)
blood_sugar = np.random.randint(80, 120, num_samples)
wbc_count = np.random.randint(4000, 8000, num_samples)
rbc_count = np.random.uniform(4.0, 6.0, num_samples)
headache = np.random.randint(0, 2, num_samples)
nausea = np.random.randint(0, 2, num_samples)
vomiting = np.random.randint(0, 2, num_samples)
seizures = np.random.randint(0, 2, num_samples)
memory_loss = np.random.randint(0, 2, num_samples)
vision_problems = np.random.randint(0, 2, num_samples)
difficulty_speaking = np.random.randint(0, 2, num_samples)
fatigue = np.random.randint(0, 2, num_samples)
balance_problems = np.random.randint(0, 2, num_samples)
coordination_difficulty = np.random.randint(0, 2, num_samples)
personality_changes = np.random.randint(0, 2, num_samples)
depression = np.random.randint(0, 2, num_samples)
anxiety = np.random.randint(0, 2, num_samples)
difficulty_concentrating = np.random.randint(0, 2, num_samples)
sleep_problems = np.random.randint(0, 2, num_samples)

# Generate the target variable randomly with a higher probability of tumor presence
tumor_present = np.random.choice([0, 1], num_samples, p=[0.1, 0.9])

# Create a DataFrame
data = {
    'Age': age,
    'Tumor_Size_cm': tumor_size,
    'Blood_Pressure_mmHg': blood_pressure,
    'Cholesterol_mg/dL': cholesterol,
    'Family_History': family_history,
    # 'BMI': bmi,
    'Hemoglobin_g/dL': hemoglobin,
    'Blood_Sugar_mg/dL': blood_sugar,
    'White_Blood_Cell_Count_K/uL': wbc_count,
    'Red_Blood_Cell_Count_million/uL': rbc_count,
    'Headache': headache,
    'Nausea': nausea,
    'Vomiting': vomiting,
    'Seizures': seizures,
    'Memory_Loss': memory_loss,
    'Vision_Problems': vision_problems,
    'Difficulty_Speaking': difficulty_speaking,
    # 'Fatigue': fatigue,
    'Balance_Problems': balance_problems,
    'Coordination_Difficulty': coordination_difficulty,
    'Personality_Changes': personality_changes,
    'Depression': depression,
    'Anxiety': anxiety,
    'Difficulty_Concentrating': difficulty_concentrating,
    # 'Sleep_Problems': sleep_problems,
    'Tumor_Present': tumor_present,
}

df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv('synthetic_tumor_dataset.csv', index=False)

# Display the first few rows of the dataset
print(df.head())
