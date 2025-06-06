import numpy as np
import copy
import multiprocessing as mp
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import sys
import os
import matplotlib.pyplot as plt

def GRR(epsilon, label_data):
    result = copy.deepcopy(label_data)
    domain = [i for i in range(min(label_data), max(label_data)+1)]
    p = np.exp(epsilon)/(np.exp(epsilon)+len(domain)-1)
    q = 1/(np.exp(epsilon)+len(domain)-1)
    sample_items = np.random.randint(1, len(domain)+1, len(label_data))
    
    sample_probs = np.random.rand(len(label_data))
    for i in range(len(label_data)):
        if sample_probs[i] > p-q:
            # print("change")   
            result[i] = sample_items[i]
    return np.array(result), domain


def OUE_perturbation(epsilon, positions, domain_length, perturbed_label, feature_frequency):
    p = 1/2
    q = 1/(np.exp(epsilon) + 1)
    # p = np.exp(epsilon)/(np.exp(epsilon) + 1)
    # q = 1/(np.exp(epsilon) + 1)

    chunck_size = 10000
    for start in range(0, len(positions), chunck_size):
        print(start)
        end = min(start+chunck_size, len(positions))
        probs = np.random.rand(end-start, domain_length+1)
        is_element = (np.arange(domain_length+1)==positions[start:end, None]).astype(int)
        vector = (probs<(p*is_element+q*(1-is_element))).astype(int)
        for i in range(end-start):
            for j in range(len(vector[i])-1):
                if vector[i][j] == 1:
                    feature_frequency[perturbed_label[start+i]][j+1] += 1
    return feature_frequency

    
def first_partition(para):
    epsilon, data, seed = para
    np.random.seed(seed)
    item_domain = [i for i in range(1, max(data[:, 0])+1)]
    label_domain = [i for i in range(1, max(data[:, 1])+1)]
    
    real_frequency = {label_value:{value: 0 for value in item_domain} for label_value in label_domain}
    for data_index in range(len(data)):
        real_frequency[data[data_index][1]][data[data_index][0]] += 1
    for label in label_domain:
        for item in item_domain:
            real_frequency[label][item] = real_frequency[label][item]
    
    perturbed_label, _ = GRR(epsilon/2, data[:, 1])

    label_frequency = {value: 0 for value in label_domain}
    for label_value in label_frequency.keys():
        label_frequency[label_value] += len(np.where(perturbed_label == label_value)[0])
    p1 = np.exp(epsilon/2)/(np.exp(epsilon/2)+len(label_frequency.keys())-1)
    q1 = 1/(np.exp(epsilon/2)+len(label_frequency.keys())-1)
    for label_value in label_frequency.keys():
        label_frequency[label_value] = (label_frequency[label_value]-len(data)*q1)/(p1-q1)
    
    positions = np.zeros(len(data))
    for data_index in range(len(data)):
        positions[data_index] = data[data_index][0]-1 
    positions = positions.astype(int)

    frequency = {label_value:{value: 0 for value in item_domain} for label_value in label_domain}
    frequency = OUE_perturbation(epsilon/2, positions, max(item_domain), perturbed_label, frequency)
    item_frequency = {value: 0 for value in item_domain}
    for label_value in frequency.keys():
        for value in frequency[label_value].keys():
            item_frequency[value] += frequency[label_value][value]
    p2 = 1/2
    q2 = 1/(np.exp(epsilon/2)+1)
    # p2 = np.exp(epsilon/4)/(np.exp(epsilon/4)+1)
    # q2 = 1/(np.exp(epsilon/4)+1)
    for value in item_frequency.keys():
        item_frequency[value] = (item_frequency[value]-len(data)*q2)/(p2-q2)
    
    for label_value in frequency.keys():
        for value in frequency[label_value].keys():
            frequency[label_value][value] = (frequency[label_value][value]-len(data)*q1*q2-label_frequency[label_value]*q2*(p1-q1)-item_frequency[value]*q1*(p2-q2))/(p1-q1)/(p2-q2)
    return real_frequency, frequency



if __name__=='__main__':
    epsilon = 1
    data = np.load('./SYN2/test_data.npy')
    seed = int(sys.argv[1])
    random_seed = 2024+seed

    item_domain = [i for i in range(1, max(data[:, 0])+1)]
    label_domain = [i for i in range(1, max(data[:, 1])+1)]
    real_frequency = {label_value:{value: 0 for value in item_domain} for label_value in label_domain}
    for data_index in range(len(data)):
        real_frequency[data[data_index][1]][data[data_index][0]] += 1
    np.save('./SYN2/OUE/real_frequency.npy', real_frequency)

    for index_ in range(1000):
        random_seed = 2024+seed+index_*10
        real_frequency, frequency = first_partition((epsilon, data, random_seed))
        np.save('./SYN2/OUE/frequency_'+str(seed+index_*10)+'-'+str(epsilon)+'.npy', frequency)
        random_seed = 2024+9+index_*10
        real_frequency, frequency = first_partition((epsilon, data, random_seed))
        np.save('./SYN2/OUE/frequency_'+str(9+index_*10)+'-'+str(epsilon)+'.npy', frequency)
    exit()

    
    