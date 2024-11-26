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


def OUE_perturbation_old(epsilon, positions, domain_length):
    np.random.seed(0)
    p = 1/2
    q = 1/(np.exp(epsilon) + 1)
    probs = np.random.rand(len(positions), domain_length+1)
    # print(probs)
    vector = [[0 for _ in range(domain_length+1)] for _ in range(len(positions))]
    for i in range(len(positions)):
        for j in range(domain_length+1):
            if positions[i][j] == 1:
                if probs[i][j]<p:
                    vector[i][j] = 1
                else:
                    vector[i][j] = 0
            elif positions[i][j] == 0:
                    if probs[i][j]>q:
                        vector[i][j] = 0
                    else:
                        vector[i][j] = 1
    return np.array(vector)

def OUE_perturbation(epsilon, positions, domain_length, perturbed_label, feature_frequency):
    p = 1/2
    q = 1/(np.exp(epsilon) + 1)
    chunck_size = 10000
    for start in range(0, len(positions), chunck_size):
        end = min(start+chunck_size, len(positions))
        probs = np.random.rand(end-start, domain_length+1)
        is_element = (np.arange(domain_length+1)==positions[start:end, None]).astype(int)
        vector = (probs<(p*is_element+q*(1-is_element))).astype(int)
        for i in range(end-start):
            if vector[i][-1] == 1:
                continue
            else:
                for j in range(len(vector[i])-1):
                    if vector[i][j] == 1:
                        feature_frequency[perturbed_label[start+i]][j+1] += 1
    return feature_frequency

def first_partition(para):
    epsilon, data, label, seed = para
    print(seed)
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

        # collect the frequency in the whole dataset
        real_feature_frequency = {label_value:{value: 0 for value in range(1, values_domain[feature_index]+1)} for label_value in labels_domain}
        for data_index in range(len(data_process)):
            real_feature_frequency[label_process[data_index]][data_process[data_index]] += 1
        
        # perturb the label
        perturbed_label, _ = GRR(epsilon/2, label_process)
        for label_value in label_frequency.keys():
            label_frequency[label_value] += len(np.where(perturbed_label == label_value)[0])
        # calibrate the labels
        p1 = np.exp(epsilon/2)/(np.exp(epsilon/2)+len(label_frequency.keys())-1)
        q1 = 1/(np.exp(epsilon/2)+len(label_frequency.keys())-1)
        for label_value in label_frequency.keys():
            label_frequency[label_value] = (label_frequency[label_value]-len(label_process)*q1)/(p1-q1)
        
        positions = np.zeros(len(data_process))
        for data_index in range(len(data_process)):
            if perturbed_label[data_index] == label_process[data_index]:
                positions[data_index] = data_process[data_index]-1 
            else:
                positions[data_index] = values_domain[feature_index]
        positions = positions.astype(int)
        feature_frequency = OUE_perturbation(epsilon/2, positions, values_domain[feature_index], perturbed_label, feature_frequency)

        p2 = 1/2
        q2 = 1/(np.exp(epsilon/2)+1)
        for label_value in feature_frequency.keys():
            for value in feature_frequency[label_value].keys():
                feature_frequency[label_value][value] = (feature_frequency[label_value][value]-len(data_process)*q1*(1-p2)*q2-label_frequency[label_value]*(p1*(1-q2)-q1*(1-p2))*q2)/p1/(1-q2)/(p2-q2)
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
        iter_nums = 20
        for i in range(iter_nums):
            para.append((epsilon, data, label, i))
    
        result = []
        for iter_ in range(iter_nums):
            result.append(first_partition(para[iter_]))
        rmse = []
        for elem in result:
            rmse.append(elem)
        print(sum(rmse)/iter_nums)
        if not os.path.exists('./result_'+task+'/result_CP_'+task):
            os.makedirs('./result_'+task+'/result_CP_'+task)
        file = open('./result_'+task+'/result_CP_'+task+'/'+str(epsilon)+'_.txt', 'w')
        file.write(str(sum(rmse)/iter_nums)+'\n')
        file.close()
    
    
