import numpy as np
import copy
import multiprocessing
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import sys
import os

def GRR(epsilon, label_data):
    result = copy.deepcopy(label_data)
    domain = [i for i in range(min(label_data), max(label_data)+1)]
    p = np.exp(epsilon)/(np.exp(epsilon)+len(domain)-1)
    q = 1/(np.exp(epsilon)+len(domain)-1)
    sample_items = np.random.randint(1, len(domain)+1, len(label_data))
    
    sample_probs = np.random.rand(len(label_data))
    for i in range(len(label_data)):
        if sample_probs[i] > p-q:
            result[i] = sample_items[i]
    return np.array(result), domain


def OUE_perturbation(epsilon, positions, domain_length):
    p = 1/2
    q = 1/(np.exp(epsilon) + 1)
    
    probs = np.random.rand(len(positions), domain_length)
    args_list = []
    for i in range(len(positions)):
        args_list.append((positions[i], probs[i], domain_length, p, q))
    vector = np.zeros((len(positions), domain_length), dtype=int)
    with multiprocessing.Pool(30) as pool:
        result = pool.starmap(OUE_perturbation_function, args_list)
    for i in range(len(positions)):
        vector[i] = result[i]
    return vector

def OUE_perturbation_function(elem, probs, domain_length, p, q):
    vector = np.zeros(domain_length, dtype=int)
    for j in range(domain_length):
        if elem[j] == 1:
            if probs[j]<p:
                vector[j] = 1
            else:
                vector[j] = 0
        elif elem[j] == 0:
            if probs[j]>q:
                vector[j] = 0
            else:
                vector[j] = 1
    return vector


def aggregate_feature_frequency(positions, perturbed_label, feature_frequency):
    label_domain = len(feature_frequency.keys())
    items_count = np.zeros((label_domain, positions.shape[1]))
    args_list = []
    for i in range(perturbed_label.shape[0]):
        args_list.append((positions[i], label_domain, perturbed_label[i]))
    with multiprocessing.Pool(30) as pool:
        result = pool.starmap(aggregate_feature_frequency_function, args_list)
    for i in range(len(result)):
        items_count += result[i]
    for i in range(label_domain):
        for j in range(len(feature_frequency[i+1])):
            feature_frequency[i+1][j+1] = items_count[i][j]
    return feature_frequency

def aggregate_feature_frequency_function(position, label_domain, label):
    items_count = np.zeros((label_domain, len(position)))
    for i in range(len(position)):
        if position[i] == 1:
            items_count[label-1][i] = 1
    return items_count
    

def first_partition(para):
    epsilon, data, label, seed = para
    np.random.seed(seed)
    feature_domain = data.shape[1]
    values_domain = {feature_index:max(data[:, feature_index]) for feature_index in range(feature_domain)}
    labels_domain = [i for i in range(min(label), max(label)+1)]

    chunk_size = len(data)//feature_domain
    data_partitions = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    label_partitions = [label[i:i+chunk_size] for i in range(0, len(label), chunk_size)]
    
    rmse = 0
    domain_amount = 0

    for feature_index in range(feature_domain):
        feature_frequency = {label_value:{value: 0 for value in range(1, values_domain[feature_index]+1)} for label_value in labels_domain}
        label_frequency = {value: 0 for value in range(min(label), max(label)+1)}

        data_process= data_partitions[feature_index][:, feature_index]
        label_process = label_partitions[feature_index]

        real_feature_frequency = {label_value:{value: 0 for value in range(1, values_domain[feature_index]+1)} for label_value in labels_domain}
        for data_index in range(len(data_process)):
            real_feature_frequency[label_process[data_index]][data_process[data_index]] += 1
        
        # perturb the label
        perturbed_label, _ = GRR(epsilon/2, label_process)
        for label_value in label_frequency.keys():
            label_frequency[label_value] += len(np.where(perturbed_label == label_value)[0])
        p1 = np.exp(epsilon/2)/(np.exp(epsilon/2)+len(label_frequency.keys())-1)
        q1 = 1/(np.exp(epsilon/2)+len(label_frequency.keys())-1)
        for label_value in label_frequency.keys():
            label_frequency[label_value] = (label_frequency[label_value]-len(label_process)*q1)/(p1-q1)
        positions = np.zeros((len(data_process), values_domain[feature_index]))
        for data_index in range(len(data_process)):
            positions[data_index][data_process[data_index]-1] = 1 
        positions = OUE_perturbation(epsilon/2, positions, values_domain[feature_index])
        feature_frequency = aggregate_feature_frequency(positions, perturbed_label, feature_frequency)
        value_frequency = {value: 0 for value in range(1, values_domain[feature_index]+1)}
        for label_value in feature_frequency.keys():
            for value in feature_frequency[label_value].keys():
                value_frequency[value] += feature_frequency[label_value][value]
        # calibrate the data
        p2 = 1/2
        q2 = 1/(np.exp(epsilon/2)+1)
        for value in value_frequency.keys():
            value_frequency[value] = (value_frequency[value]-len(data_process)*q2)/(p2-q2)
        for label_value in feature_frequency.keys():
            for value in feature_frequency[label_value].keys():
                feature_frequency[label_value][value] = (feature_frequency[label_value][value]-len(data_process)*q1*q2-label_frequency[label_value]*q2*(p1-q1)-value_frequency[value]*q1*(p2-q2))/(p1-q1)/(p2-q2)
        for label_value in real_feature_frequency.keys():
            for value in real_feature_frequency[label_value].keys():
                rmse += abs(real_feature_frequency[label_value][value]-feature_frequency[label_value][value])**2
        domain_amount += values_domain[feature_index]*len(labels_domain)
    return np.sqrt(rmse/domain_amount)


if __name__ == '__main__':
    data = np.load('./data/diabetes/data.npy')
    label = np.load('data/diabetes/label.npy')
    task = 'diabetes'
    # data = np.load('./data/heart_disease/data.npy')
    # label = np.load('./data/heart_disease/label.npy')
    # task = 'heart_disease'
    core_number = 30
    epsilons = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]

    for epsilon in epsilons:
        print("epsilon", epsilon)
        para = []
        iter_ = 20
        rmse = []
        for i in range(iter_):
            result = first_partition((epsilon, data, label, i))
            rmse.append(result)
            print(i, result)
        print(sum(rmse)/iter_)
        if not os.path.exists('./result_'+task+'/result_OUE_'+task):
            os.makedirs('./result_'+task+'/result_OUE_'+task)
        file = open('./result_'+task+'/result_OUE_'+task+'/'+str(epsilon)+'_.txt', 'w')
        file.write(str(sum(rmse)/iter_)+'\n')
        file.close()
    
    
