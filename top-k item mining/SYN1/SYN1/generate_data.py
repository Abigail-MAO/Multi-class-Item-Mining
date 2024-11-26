import numpy as np
from collections import Counter

np.random.seed(2024)

item_domain = 20000
data_amount = 5000000

mean = 0
std_dev = 2
n_parts = [10, 20, 30, 40, 50]
# n_parts = [50]
# n_parts = [10]
for n_part in n_parts:
    # 生成每部分数据量的高斯分布
    data_points = np.linspace(0, 3.5,  n_part)
    data_sizes = np.exp(-((data_points - mean) ** 2) / (2 * std_dev ** 2))
    data_sizes = data_sizes/sum(data_sizes)
    
    data_sizes_in_thousands = (data_sizes * data_amount).astype(int)
    # data_sizes_in_thousands = (data_amount/n_part*np.ones(n_part)).astype(int)
    # print(data_sizes_in_thousands)
    # exit()
    print(n_part, "总数据量:", sum(data_sizes_in_thousands))

    indices = np.arange(1, item_domain + 1)
    mapped_indicies = np.random.permutation(indices)

    # 生成和保存每部分数据

    scales = []
    start_value = 0.1
    end_value = 0.001
    x = np.linspace(0, 1, n_part)
    scales = end_value + (start_value - end_value) * np.exp(-5 * x)
    # scales = [0.05 for _ in range(n_part)]
    # print(scales)
    # exit(0)
    data = np.empty((0, 2), int)
    for part in range(len(data_sizes_in_thousands)):
        np.random.seed(part)
        data_size = data_sizes_in_thousands[part]
        
        scale =scales[part]
        # print("sale", scale)
        probs = [scale*1/(np.exp(scale*i)) for i in range(1, item_domain+1)]
        probs = np.array(probs) / sum(probs)
        # probs = np.interp(np.linspace(0, len(probs)-1, item_domain), np.arange(len(raw_probs)), raw_probs)
        # print("len probs", len(probs))

        # print(probs)
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
        # print("zero count", zero_count)
        # print(user_amount[:20])
        # print("sum", sum(user_amount))
        # print("len unique values", len(np.unique(values)))
    
        n = np.random.randint(1, 21)
        weights = np.linspace(20, 1, num=20)
        selected_indices = np.random.choice(np.arange(20), n, replace=False, p=weights/sum(weights))
        selected_indices = np.sort(selected_indices)
        # print(selected_indices)
        new_indices = [None for _ in range(20)]
        new_positions = []
        for i, index in enumerate(selected_indices):
            position_weights = np.linspace(10, 1, num=20) 
            position_probs = position_weights / position_weights.sum()
            while True:
                position = np.random.choice(np.arange(20), p=position_probs)
                if position not in new_positions:
                    new_positions.append(position)
                    new_indices[position] = mapped_indicies[selected_indices[i]]
                    break
        padded_indices = np.random.choice(np.arange(20, 50), 20-n, replace=False)
        pad_i = 0
        for i in range(20):
            if new_indices[i] is None:
                new_indices[i] = mapped_indicies[padded_indices[pad_i]]
                pad_i += 1
        new_indices = np.array(new_indices)
        remaining_indices = mapped_indicies[:]
        # print(new_indices)
        delete_indices = np.concatenate((selected_indices, padded_indices))
        remaining_indices = np.delete(remaining_indices, delete_indices)
        remaining_indices = np.random.permutation(remaining_indices)
        new_indices = np.concatenate((new_indices, remaining_indices))
        values = np.array([new_indices[i-1] for i in values])
        values = np.random.permutation(values)
        labels = np.array([part+1 for _ in range(len(values))])
        result = np.column_stack((values, labels))
        data = np.append(data, result, axis=0)
        print("len unique values", len(np.unique(values)))


    np.save('./data/Class_' + str(n_part) + '.npy', data)

        
    
    
