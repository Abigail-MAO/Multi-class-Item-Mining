import numpy as np 
import math
import gc
import os
from OUE import *
from metric import *
import multiprocessing
from collections import Counter
import time
import sys
import copy

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


def find_top_k_per_category(epsilon, data, iter_num, k, seed, candidates):
    np.random.seed(seed)
    bucket_length = 4*k
    remain_length = 2*k
    user_groups = np.linspace(0, len(data), iter_num+1, dtype=np.int32)
    for i in range(iter_num):
        print("iter num", i)
        if len(candidates)<4*k:
            bucket_length = min(len(candidates), 4*k)
            remain_length = bucket_length//2
        users_data = data[user_groups[i]:user_groups[i+1]]
        bucket, index_record = uniformly_split_array(candidates, bucket_length)
        user_position = [bucket_length for _ in range(len(users_data))]
       
        for elem_index in range(len(users_data)):
            if users_data[elem_index] in index_record.keys():
                user_position[elem_index] = index_record[users_data[elem_index]]
            else:
                user_position[elem_index] = bucket_length
        bucket_counts = modified_OUE_v2(user_position, bucket_length, epsilon)
        gc.collect()
        top_k_bucket_indicies = np.argsort(bucket_counts)[::-1][:remain_length]
        
        candidates = []
        for index in top_k_bucket_indicies:
            candidates.extend(bucket[index])
        if i == iter_num-2:
            bucket_length = len(candidates)
            remain_length = k
    return candidates

    
def multi_category_finding(file_name, epsilon, k, seed, ground_truth, file_path):
    np.random.seed(seed)
    initial_data = np.load(file_name)
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
   
    candidates = find_top_k_per_category(epsilon, data, max_iter_num, find_domain, seed, candidates)
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
    file = open(file_path+'/initial_result/'+str(epsilon)+'_'+str(top_k_value)+'_result_'+str(seed)+'.txt', 'w')
    file.write(str(f1_result)+'\n')
    file.write(str(NCG_result)+'\n')
    file.close()
    print(f1_result, NCG_result)
    return (f1_result, NCG_result)
    
        
def obtain_sorted_result():
    data = np.load(file_name)
    counter_by_group = {}
    for item, group_id in data:
        counter_by_group.setdefault(group_id, Counter())[item] += 1
    sorted_result = {group_id: [item for item, _ in counter.most_common()] for group_id, counter in counter_by_group.items()}
    return sorted_result


if __name__ == '__main__':
    # file_name = '../JD_dataset/JD_data_age_sampled.npy'
    # task = 'JD/'
    file_name = '../Anime_dataset/anime_data.npy'
    task = 'anime/'

    file_path = '../result_'+task+'PTJ_all/'
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        os.makedirs(file_path+'/initial_result')
    
    core_number = int(sys.argv[1])
    epsilon = float(sys.argv[2])
    top_k_value = int(sys.argv[3])
    
    max_process = core_number
    iter_nums = 20
    fragment_length = 2
    print("epsilon:", epsilon, "top-k:", top_k_value, file_name)
    ground_truth = ground_truth_obtain(file_name, top_k_value)
    args_list = [(file_name, epsilon, top_k_value, seed, ground_truth, file_path) for seed in range(iter_nums)]
        
    total_f1_score = 0
    total_ncg_score = 0
    iter_count = 0
    for iter_ in range(iter_nums):
        elem = multi_category_finding(file_name, epsilon, top_k_value, iter_, ground_truth, file_path)
        if not elem[0]:
            continue
        total_f1_score += elem[0]
        total_ncg_score += elem[1]
        iter_count += 1
        gc.collect()
    print(total_f1_score/iter_count)
    print(total_ncg_score/iter_count)
    
    file = open(file_path+str(epsilon)+'_'+str(top_k_value)+'_result.txt', 'w')
    file.write(str(total_f1_score/iter_count)+'\n')
    file.write(str(total_ncg_score/iter_count)+'\n')
    file.close()
    

