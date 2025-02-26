import numpy as np 


array = []
n1 = 1000000
n2 = 100000
n3 = 10000
n4 = 1000


for i in range(n1):
    array.append([1, 1])
    array.append([2, 2])
    array.append([3, 3])
    array.append([4, 4])
   
for i in range(n2):
    array.append([1, 2])
    array.append([2, 3])
    array.append([3, 4])
    array.append([4, 1])
    
for i in range(n3):
    array.append([1, 3])
    array.append([2, 4])
    array.append([3, 1])
    array.append([4, 2])

for i in range(n4):
    array.append([1, 4])
    array.append([2, 1])
    array.append([3, 2])
    array.append([4, 3])

array = np.array(array)
array = np.random.permutation(array)
np.save('./SYN1/test_data.npy', array)
items = np.unique(array[:, 0])
labels = np.unique(array[:, 1])

frequency = {label_value:{value: 0 for value in items} for label_value in labels}
for index in range(len(array)):
    frequency[array[index][1]][array[index][0]] += 1
print(frequency)