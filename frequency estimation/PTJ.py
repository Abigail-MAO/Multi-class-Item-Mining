import numpy as np
import copy
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import sys
import os

def GRR(epsilon, label_data, domain_length):
    result = copy.deepcopy(label_data)
    domain = [i for i in range(domain_length)]
    p = np.exp(epsilon)/(np.exp(epsilon)+len(domain)-1)
    q = 1/(np.exp(epsilon)+len(domain)-1)
    sample_items = np.random.randint(0, len(domain), len(label_data))
    
    sample_probs = np.random.rand(len(label_data))
    for i in range(len(label_data)):
        if sample_probs[i] > p-q:
            result[i] = sample_items[i]
    return np.array(result), domain

def OUE(epsilon, label_data, domain_length):
    p = 1/2
    q = 1/(np.exp(epsilon) + 1)
    result = np.zeros(domain_length)
    label_data = np.array(label_data)
    chunk_size = 100000
    for start in range(0, len(label_data), chunk_size):
        end = min(start+chunk_size, len(label_data))
        is_element = (np.arange(domain_length)==label_data[start:end, None]).astype(int)
        probs = np.random.rand(end-start, domain_length)
        vector = (probs<(p*is_element+q*(1-is_element))).astype(int)
        result += np.sum(vector, axis=0)
    return result.astype(int)

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

        count = 0
        label_item = {}
        inverse_label_item = {}
        for label_index in labels_domain:
            for item in range(1, values_domain[feature_index]+1):
                label_item[(label_index, item)] = count
                inverse_label_item[count] = (label_index, item)
                count += 1
        
        
        raw_data = []
        for i in range(len(data_process)):
            raw_data.append(label_item[(label_process[i], data_process[i])])
        raw_data = np.array(raw_data)

        if count<3*np.exp(epsilon)+2:
            perturbed_data, _ = GRR(epsilon, raw_data, count)
            feature_fequency = {label_value:{value: 0 for value in range(1, values_domain[feature_index]+1)} for label_value in labels_domain}
            for i in range(len(perturbed_data)):
                label_elem, item = inverse_label_item[perturbed_data[i]]
                feature_fequency[label_elem][item] += 1
            
            p = np.exp(epsilon)/(np.exp(epsilon)+values_domain[feature_index]-1)
            q = 1/(np.exp(epsilon)+values_domain[feature_index]-1)
            for label_value in feature_fequency.keys():
                for value in feature_fequency[label_value].keys():
                    feature_fequency[label_value][value] = (feature_fequency[label_value][value]-len(perturbed_data)*q)/(p-q)
        else:
            perturbed_result = OUE(epsilon, raw_data, count)
            p = 1/2
            q = 1/(np.exp(epsilon)+1)
            perturbed_result = (perturbed_result-len(raw_data)*q)/(p-q)
            feature_fequency = {label_value:{value: 0 for value in range(1, values_domain[feature_index]+1)} for label_value in labels_domain}
            for value in range(count):
                label_elem, item = inverse_label_item[value]
                feature_fequency[label_elem][item] += perturbed_result[value]
           
        real_feature_frequency = {label_value:{value: 0 for value in range(1, values_domain[feature_index]+1)} for label_value in labels_domain}
        for data_index in range(len(data_process)):
            real_feature_frequency[label_process[data_index]][data_process[data_index]] += 1
        diffs = []
        for label_value in real_feature_frequency.keys():
            for value in real_feature_frequency[label_value].keys():
                diff = (real_feature_frequency[label_value][value]-feature_fequency[label_value][value])**2
                diffs.append(diff)
                rmse += (real_feature_frequency[label_value][value]-feature_fequency[label_value][value])**2
        domain_amount += values_domain[feature_index]*len(labels_domain)
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
        iter_nums = 20
        for i in range(iter_nums):
            para.append((epsilon, data, label, i))
        result = []
        for iter_ in range(iter_nums):
            print(iter_)
            result.append(first_partition(para[iter_]))
        rmse = []
        for elem in result:
            rmse.append(elem)
        print(sum(rmse)/iter_nums)
        if not os.path.exists('./result_'+task+'/result_GRR_cal_'+task):
            os.makedirs('./result_'+task+'/result_GRR_cal_'+task)
        file = open('./result_'+task+'/result_GRR_cal_'+task+'/'+str(epsilon)+'_.txt', 'w')
        file.write(str(sum(rmse)/iter_nums)+'\n')
        file.close()
    
    
