import pandas as pd 
import numpy as np


user_anime = pd.read_csv('animelists_filtered.csv')
user_info = pd.read_csv('users_filtered.csv')
merged_data = pd.merge(user_anime, user_info, on='username')[['anime_id', 'gender']]
merged_data.to_csv('filtered_data.csv')
merged_data = pd.read_csv('filtered_data.csv')[['anime_id', 'gender']]

merged_data['gender'] = merged_data['gender'].map({'Male': 1, 'Female': 2, 'Non-Binary': None})
cleaned_data = merged_data.dropna()
data = cleaned_data.values
np.save('filtered_gender_data.npy', data)

np.random.seed(2024)
data = np.load('filtered_gender_data.npy')
data = data.astype(int)
data = np.random.permutation(data)
initial_values = np.unique(data[:, 0])
indices = [i+1 for i in range(len(np.unique(data[:,0])))]
shuffled_indices = np.random.permutation(indices)
mapping_dict = dict(zip(initial_values, shuffled_indices))
mapped_data = data.copy()
mapped_data[:, 0] = np.array([mapping_dict[val] for val in data[:, 0]])
sample_size = int(len(mapped_data)*0.2)
indices = np.random.choice(len(mapped_data), sample_size, replace=False)
mapped_data = mapped_data[indices]
np.save('anime_data.npy', mapped_data)


