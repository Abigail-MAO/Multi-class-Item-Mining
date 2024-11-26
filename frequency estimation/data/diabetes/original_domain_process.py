import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(2024)
data = pd.read_csv('diabetes.csv')
data = data.dropna()
print(len(data))
data['gender'] =data['gender'].map({'Female':1, 'Male':2, 'Other': 3})
data['age'] = np.ceil(data['age']).astype(int)
data['hypertension'] = data['hypertension']+1
data['heart_disease'] = data['heart_disease']+1
data['smoking_history'] = data['smoking_history'].map({'never':1, 'former':2, 'ever':3, 'not current':4, 'No Info':5, 'current':6})
data['bmi'] = (data['bmi']*10).astype(int)
unique_bmi_sorted = sorted(data['bmi'].unique())
bmi_mapping = {bmi:idx+1 for idx, bmi in enumerate(unique_bmi_sorted)}
data['bmi'] = data['bmi'].map(bmi_mapping)
unique_HbA1c_sorted = sorted(data['HbA1c_level'].unique())
HbA1c_mapping = {HbA1c:idx+1 for idx, HbA1c in enumerate(unique_HbA1c_sorted)}
data['HbA1c_level'] = data['HbA1c_level'].map(HbA1c_mapping)

unique_blood_glucose_sorted = sorted(data['blood_glucose_level'].unique())
blood_glucose_mapping = {blood_glucose:idx+1 for idx, blood_glucose in enumerate(unique_blood_glucose_sorted)}
data['blood_glucose_level'] = data['blood_glucose_level'].map(blood_glucose_mapping)
data['diabetes'] = data['diabetes']+1

y = data['diabetes']
X = data.copy()
X = X.drop('diabetes', axis=1)

indices = np.random.permutation([i for i in range(len(X))])
X = X.iloc[indices]
y = y.iloc[indices]

np.save('data.npy', X)
np.save('label.npy', y)
