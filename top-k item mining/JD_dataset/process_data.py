import numpy as np 
import pandas as pd 

file_names = ['JData_Action_201602.csv', 'JData_Action_201603.csv', 'JData_Action_201604.csv']
list_of_dataframes = []

for file_name in file_names:
    data = pd.read_csv(file_name)
    list_of_dataframes.append(data)
    
combined_data = pd.concat(list_of_dataframes)
combined_data.to_csv('combined_all_records.csv', index=False)

users = pd.read_csv('JData_User.csv', encoding='gbk')
users = users[users['age'] != '-1']
users = users.dropna(subset=['age'])
age_ranges = ['15岁以下', '16-25岁', '26-35岁', '36-45岁', '46-55岁', '56岁以上']
age_mapping = {
    '15岁以下': 1,
    '16-25岁': 1,
    '26-35岁': 2,
    '36-45岁': 3,
    '46-55岁': 4,
    '56岁以上': 5
}
users['age_encoded'] = users['age'].map(age_mapping)


records = pd.read_csv('combined_all_records.csv')
records['user_id'] = records['user_id'].astype(int)

merged_data = pd.merge(records, users[['user_id', 'age_encoded']], on='user_id', how='left')
merged_data = merged_data.dropna(subset=['age_encoded'])
merged_data['age_encoded'] = merged_data['age_encoded'].astype(int)

selected_columns = merged_data[['sku_id', 'age_encoded']]
selected_columns.to_csv('items_age.csv', index=False, header=False)

data = pd.read_csv('items_age.csv', header=None)
np.save('JD_data_age.npy', data.to_numpy())

np.random.seed(2024)
data =  np.load('JD_data_age.npy', allow_pickle=True)
num_samples = int(data.shape[0]*0.2)
row_indicies = np.random.choice(data.shape[0], num_samples, replace=False)
data = data[row_indicies]
np.save('JD_data_age_sampled.npy', data)



