import numpy as np 
from OUE import *
from metric import *
import multiprocessing
from collections import Counter
import time
import copy
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
        print("iter num", i)
        if len(candidates)<4*k:
            bucket_length = min(len(candidates), 4*k)
            remain_length = bucket_length//2
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
        bucket_counts = modified_OUE_v2(user_position, bucket_length, epsilon)
        top_k_bucket_indicies = np.argsort(bucket_counts)[::-1][:remain_length]
        
        candidates = []
        for index in top_k_bucket_indicies:
            candidates.extend(bucket[index])
        if i == iter_num-2:
            bucket_length = len(candidates)
            remain_length = k
    return candidates
    

def process(epsilon, k, rand_seed, file_path, ground_truth):
    np.random.seed(rand_seed)
    data = np.load(file_name)
    item_domain = max(data[:, 0])
    invalid_labels = np.zeros(len(data))
    perturbed_category_labels, label_domain = GRR(epsilon/2, data[:, 1])
    invalid_labels[perturbed_category_labels!=data[:, 1]] = 1
    data[:, 1] = perturbed_category_labels
    category, _ = np.unique(data[:, 1], return_counts=True)
    candidates = [i for i in range(1, item_domain+1)]
    UE_length = 4*k
    max_iter_num = np.ceil(np.log2(len(candidates)/UE_length))+1
    iter_nums = [max_iter_num for _ in range(len(category))]
    
    quatar = 2
    first_iter_num = int(max_iter_num/quatar)
    users_indicies = np.random.choice(len(data), int(len(data)/5), replace=False)
    users_data = data[users_indicies][:, 0]
    user_amount_estimation_data = data[users_indicies][:, 1]
    p = np.exp(epsilon/2)/(np.exp(epsilon/2)+len(category)-1)
    q = 1/(np.exp(epsilon/2)+len(category)-1)
    user_amount_estimation = {i:0 for i in category}
    for i in category:
        user_amount_estimation[i] = len(user_amount_estimation_data[user_amount_estimation_data==i])
    for i in category:
        user_amount_estimation[i] = (np.array(user_amount_estimation[i])-len(user_amount_estimation_data)*q)/(p-q)*quatar

    users_group = np.linspace(0, len(users_data), first_iter_num+1, dtype=np.int32)
    bucket_length = 4*k*len(category)
    remain_length = 2*k*len(category)
    for i in range(first_iter_num):
        user_iter_data = users_data[users_group[i]:users_group[i+1]]
        bucket, index_record = uniformly_split_array(candidates, bucket_length)
        user_position = [bucket_length for _ in range(len(user_iter_data))]
       
        for elem_index in range(len(user_iter_data)):
            if user_iter_data[elem_index] in index_record.keys():
                user_position[elem_index] = index_record[user_iter_data[elem_index]]
            else:
                user_position[elem_index] = bucket_length
        bucket_counts = modified_OUE_v2(user_position, bucket_length, epsilon/2)
        top_k_bucket_indicies = np.argsort(bucket_counts)[::-1][:remain_length]
        candidates = []
        for index in top_k_bucket_indicies:
            candidates.extend(bucket[index])
    data = np.delete(data, users_indicies, axis=0)
    invalid_labels = np.delete(invalid_labels, users_indicies)
    iter_nums = [max_iter_num-first_iter_num for _ in range(len(category))]
    for i in category:
        user_amount_per_category = len(data[data[:, 1]==i])
        if user_amount_per_category>user_amount_estimation[i]*2:
            invalid_labels[data[:, 1]==i] = 0
    grouped_data = [data[data[:, 1]==label][:, 0] for label in category]
    grouped_invalid_labels = [invalid_labels[data[:, 1]==label] for label in category]
    results = {}
    for index in range(len(iter_nums)):
        results[category[index]] = find_top_k_per_category(epsilon/2, grouped_data[index], candidates, int(iter_nums[index]), k, grouped_invalid_labels[index], category[index])
    f1_score_ = f1_score(results, ground_truth)
    ncg_score_ = NCG_score(ground_truth, results, k)
    file_result = open(file_path+'/initial_result/'+str(epsilon)+'_'+str(k)+'_'+str(rand_seed)+'_result.txt', 'w')
    file_result.write(str(f1_score_)+'\n')
    file_result.write(str(ncg_score_)+'\n')
    file_result.close()
    return f1_score_, ncg_score_


if __name__=='__main__':
    # file_name = '../JD_dataset/JD_data_age_sampled.npy'
    # task = 'JD/'
    file_name = '../Anime_dataset/anime_data.npy'
    task = 'anime/'
    file_path = '../result_'+task+'PTS_all/'
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
    total_f1_score = 0
    total_ncg_score = 0
    iter_count = 0
    for iter_ in range(iter_nums):
        result = process(epsilon, top_k_value, iter_, file_path, ground_truth)
        # exit()
        if result[0] == 0:
            continue
        total_f1_score += result[0]
        total_ncg_score += result[1]
        iter_count += 1
    print(total_f1_score/iter_count)
    print(total_ncg_score/iter_count)
   
    file = open(file_path+'/'+str(epsilon)+'_'+str(top_k_value)+'_result.txt', 'w')
    file.write(str(total_f1_score/iter_count)+'\n')
    file.write(str(total_ncg_score/iter_count)+'\n')
    file.close()
    




   