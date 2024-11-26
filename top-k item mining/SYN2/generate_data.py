import numpy as np
from collections import Counter

np.random.seed(2024)

item_domain = 20000
data_amount = 5000000

mean = 0
std_dev = 2
n_parts = [10, 20, 30, 40, 50]
for n_part in n_parts:
    data_points = np.linspace(0, 3.5,  n_part)
    data_sizes = np.exp(-((data_points - mean) ** 2) / (2 * std_dev ** 2))
    data_sizes = data_sizes/sum(data_sizes)
    
    data_sizes_in_thousands = (data_sizes * data_amount).astype(int)
    print(n_part, "data amount:", sum(data_sizes_in_thousands))

    indices = np.arange(1, item_domain + 1)
    mapped_indicies = np.random.permutation(indices)

    scales = []
    start_value = 0.1
    end_value = 0.001
    x = np.linspace(0, 1, n_part)
    scales = end_value + (start_value - end_value) * np.exp(-5 * x)
    data = np.empty((0, 2), int)
    for part in range(len(data_sizes_in_thousands)):
        np.random.seed(part)
        data_size = data_sizes_in_thousands[part]
        scale =scales[part]
        probs = [scale*1/(np.exp(scale*i)) for i in range(1, item_domain+1)]
        probs = np.array(probs) / sum(probs)
        values = []
        user_amount = []
        zero_count = 0
        for prob_index in range(len(probs)):
            if data_size*probs[prob_index]<1:
                zero_count += 1
            if data_size*probs[prob_index]<1:
                values.append(prob_index+1)
            for _ in range(int(data_size*probs[prob_index])):
                values.append(prob_index+1)
            user_amount.append(int(data_size*probs[prob_index]))
        new_indices = np.random.permutation(indices[:])
        values = np.array([new_indices[i-1] for i in values])
        values = np.random.permutation(values)
        labels = np.array([part+1 for _ in range(len(values))])
        result = np.column_stack((values, labels))
        data = np.append(data, result, axis=0)
        print("len unique values", len(np.unique(values)))

    np.save('./data/Class_' + str(n_part) + '.npy', data)

        
    
    
