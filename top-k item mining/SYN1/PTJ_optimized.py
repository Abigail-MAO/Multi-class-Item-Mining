import numpy as np 
import math
import gc
import os
# from OUE import *
from metric import *
import multiprocessing
from collections import Counter
import time
import sys
import copy


def modified_OUE_v2(items, domain_length, epsilon, class_num):
    p = 1/2
    q = 1 / (np.exp(epsilon) + 1)
    
    items = np.array(items)
    item_counts = np.zeros(domain_length, dtype=int)
    chunck_size = 100000
    for start in range(0, len(items), chunck_size):
        end = min(len(items), start + chunck_size)
        probs = np.random.rand(end - start, domain_length + 1)
        is_elem_position = (np.arange(domain_length + 1) == items[start:end, None])
        vector = (probs < (p * is_elem_position + q * ~is_elem_position)).astype(int)
        vector[vector[:, -1] == 1] = 0
        item_counts += np.sum(vector[:, :-1], axis=0)
    return item_counts

    
def uniformly_split_array(arr, n):
    part_size = len(arr) // n
    if part_size<1:
        result = {i:arr[i] for i in range(len(arr))}
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


def find_top_k_per_category(epsilon, data, iter_num, k, seed, candidates, class_num):
    np.random.seed(seed)
    bucket_length = 4*k
    remain_length = 2*k
    user_groups = np.linspace(0, len(data), iter_num+1, dtype=np.int32)
    for i in range(iter_num):
        print("iter num", i)
        # arrange the items into buckets and get the corresponding positions
        users_data = data[user_groups[i]:user_groups[i+1]]
        print("split candidates")
        bucket, index_record = uniformly_split_array(candidates, bucket_length)
        user_position = [bucket_length for _ in range(len(users_data))]
        print("user positions")
        for elem_index in range(len(users_data)):
            if users_data[elem_index] in index_record.keys():
                user_position[elem_index] = index_record[users_data[elem_index]]
            else:
                user_position[elem_index] = bucket_length
        print("perturb the bucket indices")
        # perturb the bucket indices
        bucket_counts = modified_OUE_v2(user_position, bucket_length, epsilon, class_num)
        gc.collect()
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

    
def multi_category_finding(data, epsilon, k, seed, ground_truth, file_path, class_num):
    np.random.seed(seed)
    initial_data = copy.deepcopy(data)
    item_domain = max(initial_data[:, 0])
    label_domain = max(initial_data[:, 1])
    elem_domain = {}
    elem_inverse = {}
    count = 1
    for i in range(1, label_domain+1):
        for j in range(1, item_domain+1):
            elem_domain[count] = (i, j)
            elem_inverse[(i, j)] = count
            count += 1
    domain_size = len(elem_domain)
    data = [elem_inverse[(int(initial_data[i][1]), int(initial_data[i][0]))] for i in range(len(initial_data))]
    candidates = [i for i in range(1, domain_size+1)]
    find_domain = label_domain*k
    UE_length = find_domain*2
    max_iter_num = int(np.ceil(np.log2(len(candidates)/UE_length))+1)
    print(max_iter_num)
   
    candidates = find_top_k_per_category(epsilon, data, max_iter_num, find_domain, seed, candidates, class_num)
    result = {i:[] for i in range(1, label_domain+1)}
    for elem in candidates:
        if elem not in elem_domain:
            continue
        label, item = elem_domain[elem]
        if len(result[label]) < k:
            result[label].append(item)
    for i in result.keys():
        if len(result[i]) < k:
            result[i].extend([None]*(k-len(result[i])))
    print(result)
    
    
    f1_result = f1_score(result, ground_truth)
    NCG_result = NCG_score(ground_truth, result, top_k_value)
    file = open(file_path+'/initial_result/'+str(class_num)+'_'+str(seed)+'_result.txt', 'w')
    file.write(str(f1_result)+'\n')
    file.write(str(NCG_result)+'\n')
    file.close()
    print(f1_result, NCG_result)
    return (f1_result, NCG_result)
    

if __name__ == '__main__':
    task = 'SYN2/'
    
    file_path = '../result_'+task+'PTJ_optimized/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        os.makedirs(file_path+'/initial_result')
   
    core_number = 30
    epsilon = 4
    top_k_value = 20
    max_process = core_number
    iter_nums = 20
    
    class_nums = [i for i in range(10, 55, 10)]
    
    for class_num in class_nums:
        total_f1_score = 0
        total_ncg_score = 0
        iter_count = 0
        
        for iter_ in range(iter_nums):
            np.random.seed(iter_)
            data = np.load('./data/Class_' + str(class_num) + '.npy', allow_pickle=True)
            data = np.random.permutation(data)
            ground_truth = ground_truth_obtain(data, top_k_value)
            elem = multi_category_finding(data, epsilon, top_k_value, iter_, ground_truth, file_path, class_num)
            if not elem[0]:
                continue
            total_f1_score += elem[0]
            total_ncg_score += elem[1]
            iter_count += 1
            gc.collect()
        print(total_f1_score/iter_count)
        print(total_ncg_score/iter_count)
    
        file = open(file_path+'/'+str(class_num)+'_'+'_result.txt', 'w')
        file.write(str(total_f1_score/iter_count)+'\n')
        file.write(str(total_ncg_score/iter_count)+'\n')
        file.close()
        

    
    

