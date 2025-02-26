import numpy as np 
from OUE import *
from metric import *
import multiprocessing
from collections import Counter
import time
import copy
import sys
import os
import gc
import psutil

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

def modified_OUE(items, domain_length, epsilon):
    p = 1/2
    q = 1/(np.exp(epsilon) + 1)
    # p = np.exp(epsilon/2)/(np.exp(epsilon/2) + 1)
    # q = 1/(np.exp(epsilon/2) + 1)
    items = np.array(items)
    probs = np.random.rand(len(items), domain_length+1)
    item_counts = np.zeros(domain_length, dtype=np.int32)
    chunk_size = 100000
    for start in range(0, len(items), chunk_size):
        end = min(len(items), start+chunk_size)
        is_elem_position = (np.arange(domain_length+1) == items[start:end, None])
        vector = (probs[start:end] < (p*is_elem_position + q*~is_elem_position)).astype(int)
        vector[vector[:, -1] == 1] = 0
        item_counts += np.sum(vector[:, :-1], axis=0)
    return item_counts


def uniformly_split_array(arr, n):
    part_size = len(arr) // n
    if part_size<1:
        result = {i:[arr[i]] for i in range(len(arr))}
        index_record = {arr[i]:i for i in range(len(arr))}
        return result, index_record
    arr = np.random.permutation(arr)
    result = {}
    index_record = {i:-1 for i in arr}
    for i in range(n):
        start = i * part_size
        end = (i + 1) * part_size if i < n - 1 else len(arr)
        result[i] = arr[start:end]
        for elem in arr[start:end]:
            index_record[elem] = i
    return result, index_record


def find_top_k_per_category(epsilon, data, candidates, iter_num, k, invalid_labels, label):
    bucket_length = 4*k
    remain_length = 2*k
    user_groups = np.linspace(0, len(data), iter_num+1, dtype=np.int32)
    for i in range(iter_num):
        # arrange the items into buckets and get the corresponding positions
        users_data = data[user_groups[i]:user_groups[i+1]]
        bucket, index_record = uniformly_split_array(candidates, bucket_length)
        user_position = [bucket_length for _ in range(len(users_data))]
        user_invalid_labels = invalid_labels[user_groups[i]:user_groups[i+1]]
        
        if iter_num-i <= 1:
            for elem_index in range(len(users_data)):
                if users_data[elem_index] in index_record.keys() and user_invalid_labels[elem_index] == 0:
                    user_position[elem_index] = index_record[users_data[elem_index]]
                else:
                    user_position[elem_index] = bucket_length
        else:
            for elem_index in range(len(users_data)):
                if users_data[elem_index] in index_record.keys():
                    user_position[elem_index] = index_record[users_data[elem_index]]
                else:
                    user_position[elem_index] = bucket_length
        
        # perturb the bucket indices
        bucket_counts = modified_OUE(user_position, bucket_length, epsilon)

        # remain the top-2k buckets
        top_k_bucket_indicies = np.argsort(bucket_counts)[::-1][:remain_length]
        
        candidates = []
        for index in top_k_bucket_indicies:
            candidates.extend(bucket[index])
        
        # the last iteration
        if i == iter_num-2:
            bucket_length = len(candidates)
            remain_length = k
    return candidates
    

def process(file_name, epsilon, k, rand_seed, file_path, ground_truth, proportion):
    np.random.seed(rand_seed)
    data = np.load(file_name)
    data = np.random.permutation(data)
    item_domain = max(data[:, 0])

    
    # perturb the category labels, if the perturbed label is not the same as its original one, the item will be assigned as -1
    invalid_labels = np.zeros(len(data))
    epsilon_1 = epsilon*proportion
    epsilon_2 = epsilon-epsilon_1
    perturbed_category_labels, label_domain = GRR(epsilon_1, data[:, 1])
    invalid_labels[perturbed_category_labels!=data[:, 1]] = 1
    data[:, 1] = perturbed_category_labels

    # get the iteration nums for each category
    category, _ = np.unique(data[:, 1], return_counts=True)
    candidates = [i for i in range(1, item_domain+1)]
    UE_length = 4*k
    max_iter_num = np.ceil(np.log2(len(candidates)/UE_length))+1
    iter_nums = [max_iter_num for _ in range(len(category))]
    
    grouped_data = [data[data[:, 1]==label][:, 0] for label in category]
    grouped_invalid_labels = [invalid_labels[data[:, 1]==label] for label in category]
    
    results = {}
    for index in range(len(iter_nums)):
        results[category[index]] = find_top_k_per_category(epsilon_2, grouped_data[index], candidates, int(iter_nums[index]), k, grouped_invalid_labels[index], category[index])
    
    f1_score_ = f1_score(results, ground_truth)
    ncg_score_ = NCG_score(ground_truth, results, k)
    print("f1_score:", f1_score_, "ncg_score:", ncg_score_)
    file_result = open(file_path+'/initial_result/'+str(epsilon)+'_'+str(k)+'_'+str(proportion)+'_'+str(rand_seed)+'_result.txt', 'w')
    file_result.write(str(f1_score_)+'\n')
    file_result.write(str(ncg_score_)+'\n')
    file_result.close()
   
    return f1_score_, ncg_score_

if __name__=='__main__':
    # generate the class 5, 10, 20 data using ./generate_data.py
    file_name1 = 'Class_10.npy'
    task1 = 'Class_10/'
    file_name2 = 'Class_5.npy'
    task2 = 'Class_5/'
    file_name3 = 'Class_20.npy'
    task3 = 'Class_20/'

    name_arrays = [file_name1, file_name2, file_name3]
    task_arrays = [task1, task2, task3]

    for index_ in range(len(name_arrays)):
        file_name = name_arrays[index_]
        task = task_arrays[index_]
        file_path = '../result/result_'+task
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            os.makedirs(file_path+'/initial_result')
        
        core_number = int(sys.argv[1])
        epsilon = float(sys.argv[2])
        top_k_value = int(sys.argv[3])
        proportion = float(sys.argv[4])
        max_process = core_number
        iter_nums = 20
        print("epsilon:", epsilon, "top-k:", top_k_value, file_name)
        ground_truth = ground_truth_obtain(file_name, top_k_value)

        total_f1_score = 0
        total_ncg_score = 0
        iter_count = 0
        for iter_ in range(iter_nums):
            result = process(file_name, epsilon, top_k_value, iter_, file_path, ground_truth, proportion)
            # exit()
            if result[0] == 0:
                continue
            total_f1_score += result[0]
            total_ncg_score += result[1]
            iter_count += 1
        print(total_f1_score/iter_count)
        print(total_ncg_score/iter_count)
    
        file = open(file_path+'/'+str(epsilon)+'_'+str(top_k_value)+'_'+str(proportion)+'_result.txt', 'w')
        file.write(str(total_f1_score/iter_count)+'\n')
        file.write(str(total_ncg_score/iter_count)+'\n')
        file.close()




   