import numpy as np 


frequency = {1:{1:10000, 2:1000000, 3:1000000, 4:1000000}, 2:{1:10000, 2:1000000, 3:100000, 4:100000}, 3:{1:10000, 2:100000, 3:100000, 4:1000}, 4:{1:10000, 2:1000, 3:1000, 4:1000}}
array = []
for label_value in frequency.keys():
    for item_value in frequency[1].keys():
        array.extend([[item_value, label_value] for _ in range(frequency[label_value][item_value])])
array = np.array(array)
array = np.random.permutation(array)
np.save('./SYN2/test_data.npy', array)
items = np.unique(array[:, 0])
labels = np.unique(array[:, 1])

frequency = {label_value:{value: 0 for value in items} for label_value in labels}
for index in range(len(array)):
    frequency[array[index][1]][array[index][0]] += 1
print(frequency)