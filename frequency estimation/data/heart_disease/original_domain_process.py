import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

data = pd.read_csv('heart_disease.csv')
data = data.dropna()

data['HeartDiseaseorAttack'] = data['HeartDiseaseorAttack']+1
data['HeartDiseaseorAttack'] = data['HeartDiseaseorAttack'].astype(int)
data['HighBP'] = data['HighBP']+1
data['HighBP'] = data['HighBP'].astype(int)
data['HighChol'] = data['HighChol']+1
data['HighChol'] = data['HighChol'].astype(int)
data['CholCheck'] = data['CholCheck']+1
data['CholCheck'] = data['CholCheck'].astype(int)
unique_BMI_sorted = sorted(data['BMI'].unique())
BMI_mapping = {BMI:idx+1 for idx, BMI in enumerate(unique_BMI_sorted)}
data['BMI'] = data['BMI'].map(BMI_mapping)
data['Smoker'] = data['Smoker']+1
data['Smoker'] = data['Smoker'].astype(int)
data['Stroke'] = data['Stroke']+1
data['Stroke'] = data['Stroke'].astype(int)
data['Diabetes'] = data['Diabetes']+1
data['Diabetes'] = data['Diabetes'].astype(int)
data['PhysActivity'] = data['PhysActivity']+1
data['PhysActivity'] = data['PhysActivity'].astype(int)
data['Fruits'] = data['Fruits'] + 1
data['Fruits'] = data['Fruits'].astype(int)
data['Veggies'] = data['Veggies'] + 1
data['Veggies'] = data['Veggies'].astype(int)
data['HvyAlcoholConsump'] = data['HvyAlcoholConsump'] + 1
data['HvyAlcoholConsump'] = data['HvyAlcoholConsump'].astype(int)
data['AnyHealthcare'] = data['AnyHealthcare']+1
data['AnyHealthcare'] = data['AnyHealthcare'].astype(int)
data['NoDocbcCost'] = data['NoDocbcCost']+1
data['NoDocbcCost'] = data['NoDocbcCost'].astype(int)
data['GenHlth'] = data['GenHlth'].astype(int)
unique_MentHlth_sorted = sorted(data['MentHlth'].unique())
MentHlth_mapping = {Hlth:idx+1 for idx, Hlth in enumerate(unique_MentHlth_sorted)}
data['MentHlth'] = data['MentHlth'].map(MentHlth_mapping)
unique_PhysHlth_sorted = sorted(data['PhysHlth'].unique())
PhysHlth_mapping = {Hlth:idx+1 for idx, Hlth in enumerate(unique_PhysHlth_sorted)}
data['PhysHlth'] = data['PhysHlth'].map(PhysHlth_mapping)
data['DiffWalk'] = data['DiffWalk']+1
data['DiffWalk'] = data['DiffWalk'].astype(int)
data['Sex'] = data['Sex']+1
data['Sex'] = data['Sex'].astype(int)
unique_Age_sorted = sorted(data['Age'].unique())
Age_mapping = {age:idx+1 for idx, age in enumerate(unique_Age_sorted)}
data['Age'] = data['Age'].map(Age_mapping)
unique_Education_sorted = sorted(data['Education'].unique())
Education_mapping = {edu:idx+1 for idx, edu in enumerate(unique_Education_sorted)}
data['Education'] = data['Education'].map(Education_mapping)
unique_Income_sorted = sorted(data['Income'].unique())
Income_mapping = {income:idx+1 for idx, income in enumerate(unique_Income_sorted)}
data['Income'] = data['Income'].map(Income_mapping)

y = data['HeartDiseaseorAttack']
X = data.copy()
X = X.drop('HeartDiseaseorAttack', axis=1)
indices = np.random.permutation([i for i in range(len(X))])
X = X.iloc[indices]
y = y.iloc[indices]
np.save('data.npy', X)
np.save('label.npy', y)

