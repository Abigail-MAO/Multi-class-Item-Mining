import numpy as np 
from OUE import *
from metric import *
import multiprocessing
from collections import Counter
import time
import copy
import sys
import math
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


def find_top_k_per_category(epsilon, data, candidates, iter_num, k, core_number):
    bucket_length = 4*k
    remain_length = 2*k
    user_groups = np.linspace(0, len(data), iter_num+1, dtype=np.int32)
    for i in range(iter_num):
        users_data = data[user_groups[i]:user_groups[i+1]]
        bucket, index_record = uniformly_split_array(candidates, bucket_length)
        user_position = np.random.randint(bucket_length, size=len(users_data))
       
        for elem_index in range(len(users_data)):
            if users_data[elem_index] in index_record.keys():
                user_position[elem_index] = index_record[users_data[elem_index]]
        bucket_counts = initial_OUE(user_position, bucket_length, epsilon, core_number)
        top_k_bucket_indicies = np.argsort(bucket_counts)[::-1][:remain_length]
        
        candidates = []
        for index in top_k_bucket_indicies:
            candidates.extend(bucket[index])
        if i == iter_num-2:
            bucket_length = len(candidates)
            remain_length = k
    return candidates
    

def process(epsilon, k, rand_seed, file_path, ground_truth, core_number):
    np.random.seed(rand_seed)
    data = np.load(file_name)
    item_domain = max(data[:, 0])
    perturbed_category_labels, _ = GRR(epsilon/2, data[:, 1])
    data[:, 1] = perturbed_category_labels
    category, _ = np.unique(data[:, 1], return_counts=True)
    candidates = [i for i in range(1, item_domain+1)]
    UE_length = 100
    max_iter_num = np.ceil(np.log2(len(candidates)/UE_length))+1
    iter_nums = [max_iter_num for _ in range(len(category))]
    grouped_data = [data[data[:, 1]==label][:, 0] for label in category]
    
    results = {}
    for index in range(len(iter_nums)):
        print(index)
        results[category[index]] = find_top_k_per_category(epsilon/2, grouped_data[index], candidates, int(iter_nums[index]), k, core_number)
    
    f1_score_ = f1_score(results, ground_truth)
    ncg_score_ = NCG_score(ground_truth, results, k)
    file_result = open(file_path+'/initial_result/'+str(epsilon)+'_'+str(k)+'_'+str(rand_seed)+'_result.txt', 'w')
    file_result.write(str(f1_score_)+'\n')
    file_result.write(str(ncg_score_)+'\n')
    file_result.close()
    print(f1_score_, ncg_score_)
    return f1_score_, ncg_score_

def obtain_sorted_result():
    data = np.load(file_name)
    counter_by_group = {}
    for item, group_id in data:
        counter_by_group.setdefault(group_id, Counter())[item] += 1
    sorted_result = {group_id: [item for item, _ in counter.most_common()] for group_id, counter in counter_by_group.items()}
    return sorted_result

if __name__=='__main__':
    # file_name = '../JD_dataset/JD_data_age_sampled.npy'
    # task = 'JD/'
    file_name = '../Anime_dataset/anime_data.npy'
    task = 'anime/'

    file_path = '../result_'+task+'PTS_shuffling/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        os.makedirs(file_path+'/initial_result')

    core_number = int(sys.argv[1])
    epsilon = float(sys.argv[2])
    top_k_value = int(sys.argv[3])
    max_process = core_number
    iter_nums = 20
    print("epsilon:", epsilon, "top-k:", top_k_value, file_name)

    ground_truth = ground_truth_obtain(file_name, top_k_value)
    args_list = [(epsilon, top_k_value, seed, file_path, ground_truth) for seed in range(iter_nums)]
    total_f1_score = 0
    total_ncg_score = 0
    iter_count = 0
    for iter_ in range(iter_nums):
        result = process(epsilon, top_k_value, iter_, file_path, ground_truth, core_number)
        if not result[0]:
            continue
        total_f1_score += result[0]
        total_ncg_score += result[1]
        iter_count += 1
    print(total_f1_score/iter_count)
    print(total_ncg_score/iter_count)
   
    file = open(file_path+str(epsilon)+'_'+str(top_k_value)+'_result.txt', 'w')
    file.write(str(total_f1_score/iter_count)+'\n')
    file.write(str(total_ncg_score/iter_count)+'\n')
    file.close()
    




   