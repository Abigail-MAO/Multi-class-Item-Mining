import numpy as np
import copy
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import sys
import os
import time


def GRR(epsilon, label_data, domain_length):
    result = copy.deepcopy(label_data)
    domain = [i for i in range(1, domain_length+1)]
    p = np.exp(epsilon)/(np.exp(epsilon)+len(domain)-1)
    q = 1/(np.exp(epsilon)+len(domain)-1)
    sample_items = np.random.randint(1, len(domain)+1, len(label_data))

    
    sample_probs = np.random.rand(len(label_data))
    for i in range(len(label_data)):
        if sample_probs[i] > p-q:
            result[i] = sample_items[i]
    return np.array(result), domain

def OUE(epsilon, label_data, domain_length):
    p = 1/2
    q = 1/(np.exp(epsilon) + 1)
    probs = np.random.rand(len(label_data), domain_length)
    vector = [[0 for _ in range(domain_length)] for _ in range(len(label_data))]
    for i in range(len(label_data)):
        for j in range(domain_length):
            if label_data[i] == j:
                if probs[i][j]<p:
                    vector[i][j] = 1
                else:
                    vector[i][j] = 0
            else:
                if probs[i][j]>q:
                    vector[i][j] = 0
                else:
                    vector[i][j] = 1
    return np.array(vector)


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
        data_process= data_partitions[feature_index][:, feature_index]
        label_process = label_partitions[feature_index]
        real_feature_frequency = {label_value:{value: 0 for value in range(1, values_domain[feature_index]+1)} for label_value in labels_domain}
        for data_index in range(len(data_process)):
            real_feature_frequency[label_process[data_index]][data_process[data_index]] += 1

        chunk_size = len(data_process)//len(labels_domain)
        label_data_process = [data_process[i:i+chunk_size] for i in range(0, len(data_process), chunk_size)]
        label_label_process = [label_process[i:i+chunk_size] for i in range(0, len(label_process), chunk_size)]

        for label_index in range(len(labels_domain)):
            value_data_process = label_data_process[label_index]
            value_label_process = label_label_process[label_index]

            elem_domain = [i for i in range(1, values_domain[feature_index]+2)]
            raw_data = []
            for i in range(len(value_data_process)):
                if value_label_process[i] == labels_domain[label_index]:
                    raw_data.append(value_data_process[i])
                else:
                    raw_data.append(elem_domain[-1])

            if len(elem_domain)<3*np.exp(epsilon)+2:
                perturbed_data, _ = GRR(epsilon, raw_data, len(elem_domain))
                feature_fequency = {label_value:{value: 0 for value in range(1, values_domain[feature_index]+1)} for label_value in labels_domain}
                for i in range(len(perturbed_data)):
                    if perturbed_data[i] == elem_domain[-1]:
                        continue
                    feature_fequency[labels_domain[label_index]][perturbed_data[i]] += 1
                
                p = np.exp(epsilon)/(np.exp(epsilon)+values_domain[feature_index]-1)
                q = 1/(np.exp(epsilon)+values_domain[feature_index]-1)
                for label_value in feature_fequency.keys():
                    for value in feature_fequency[label_value].keys():
                        feature_fequency[label_value][value] = (feature_fequency[label_value][value]-len(perturbed_data)*q)/(p-q)
            else:
                perturbed_data = OUE(epsilon, raw_data, len(elem_domain))
                feature_fequency = {label_value:{value: 0 for value in range(1, values_domain[feature_index]+1)} for label_value in labels_domain}
                for i in range(len(perturbed_data)):
                    for value in range(1, values_domain[feature_index]+1):
                        if perturbed_data[i][value-1] == 1:
                            feature_fequency[labels_domain[label_index]][value] += 1
                p = 1/2
                q = 1/(np.exp(epsilon)+1)
                for label_value in feature_fequency.keys():
                    for value in feature_fequency[label_value].keys():
                        feature_fequency[label_value][value] = (feature_fequency[label_value][value]-len(perturbed_data)*q)/(p-q)

            for label_value in real_feature_frequency.keys():
                for value in real_feature_frequency[label_value].keys():
                    rmse += (real_feature_frequency[label_value][value]-feature_fequency[label_value][value]*feature_domain)**2
                    domain_amount += 1
    return np.sqrt(rmse/domain_amount)


if __name__ == '__main__':
    data = np.load('./data/diabetes/data.npy')
    label = np.load('data/diabetes/label.npy')
    task = 'diabetes'
    # data = np.load('./data/heart_disease/data.npy')
    # label = np.load('./data/heart_disease/label.npy')
    # task = 'heart_disease'

    core_number = 10
    epsilons = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]

    for epsilon in epsilons:
        print("epsilon", epsilon)
   
        para = []
        iter_ = 20
        for i in range(iter_):
            para.append((epsilon, data, label, i))
    
        with Pool(core_number) as pool:
            result = pool.map(first_partition, para)
        rmse = []
        for elem in result:
            rmse.append(elem)
        print(sum(rmse)/iter_)
        if not os.path.exists('./result_'+task+'/result_HEC_'+task):
            os.makedirs('./result_'+task+'/result_HEC_'+task)
        file = open('./result_'+task+'/result_HEC_'+task+'/'+str(epsilon)+'_.txt', 'w')
        file.write(str(sum(rmse)/iter_)+'\n')
        file.close()
    
    
