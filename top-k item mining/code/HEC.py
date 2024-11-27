import numpy as np 
import math
import gc
import os
from metric import *
import multiprocessing
from collections import Counter
import time
import sys
import copy


def OUE_initial_perturb(epsilon, data, user_amount):
    core_number = 30
    p = 1/2
    q = 1/(np.exp(epsilon) + 1)

    sample_invalid = np.random.randint(0, len(data), data[-1])
    for invalid_item in sample_invalid:
        data[invalid_item] += 1
    bucket_length = len(data)-1
    
    item_count = np.zeros(bucket_length, dtype=np.uint32)
    sample_probs = np.random.rand(user_amount, bucket_length)
    args_list = []
    user_index = 0
    for bucket_index in range(bucket_length):
        for _ in range(data[bucket_index]):
            args_list.append((bucket_index, bucket_length, sample_probs[user_index], p, q))
            user_index += 1
    chunk_size = int(math.ceil(len(args_list)/core_number))
    with multiprocessing.Pool(processes=core_number) as pool:
        result = pool.starmap(OUE_function, args_list, chunksize=chunk_size)
    for i in result:
        item_count += i
    return item_count

def OUE_function(elem, bucket_length, probs, p, q):
    count = np.zeros(bucket_length, dtype=np.uint32)
    for bucket_index in range(bucket_length):
        if elem == bucket_index:
            if probs[bucket_index] < p:    
                count[bucket_index] += 1
        else:
            if probs[bucket_index] < q:
                count[bucket_index] += 1
    return count


def suffix_match(data, candidates, suffix_length, label):
    args_list = [(candidates, data[user_index], suffix_length, label) for user_index in range(len(data))]
    with multiprocessing.Pool(processes=30) as pool:
            uploading = pool.starmap(suffix_match_function, args_list)
    can_count = np.zeros(len(candidates)+1, dtype=np.uint32)
    
    for elem, find in uploading:
        if find:
            can_count[elem] += 1
        else:
            can_count[-1] += 1
    return can_count, candidates


def suffix_match_function(candidates, elem, suffix_length, label):
    mask = (1 << suffix_length) - 1
    find = False
    user_data = elem[0]
    user_label = elem[1]
    if user_label != label:
        return user_data, False
    for i in range(len(candidates)):
        if user_data & mask == candidates[i]:
            find = True
            break
    if find:
        return i, True
    else:
        return user_data, False


def generate_m_bits_candidates(candidates, start_bit, end_bit):
    bin_candidates = []
    for num in candidates:
        binary_str = bin(num)[2:]
        bin_candidates.append('0'*(start_bit-len(binary_str))+binary_str)
    padding_variations = 2**(end_bit-start_bit)
    new_candidates = []
    for i in range(padding_variations):
        padding = bin(i)[2:].zfill(end_bit-start_bit)
        for j in range(len(bin_candidates)):
            # print(padding, bin_candidates[j], padding+bin_candidates[j])
            new_candidates.append(int(padding+bin_candidates[j], 2))
    return    new_candidates 


def find_top_k_per_category(epsilon, data, max_string_length, fragment_length, k, seed, label):
    start_length = int(math.log2(k))+1
    num_iterations = 1+int(math.ceil((max_string_length-start_length)/fragment_length))
    data = np.random.permutation(data)
    groups = np.linspace(0, len(data), num_iterations+1, dtype=np.int32)
    for i in range(num_iterations):
        previous_length = start_length+(i-1)*fragment_length
        if previous_length<=1 or i==0:
            previous_length = 1
            candidates = [0, 1]
        suffix_length = start_length + i*fragment_length
        candidates = generate_m_bits_candidates(candidates, previous_length, suffix_length)
        
        users_data = data[groups[i]:groups[i+1]]
        real_stat, _ = suffix_match(users_data, candidates, suffix_length, label)
        perturbed_est = OUE_initial_perturb(epsilon, real_stat, len(users_data))
        gc.collect()
        indicies = np.argsort(perturbed_est)[::-1][:k]
        candidates = [candidates[i] for i in indicies]
    return candidates


def GRR(epsilon, data, seed):
    np.random.seed(seed)
    result = copy.deepcopy(data)
    domain = [i for i in range(min(result), max(result)+1)]
    p = np.exp(epsilon)/(np.exp(epsilon)+len(domain)-1)
    q = 1/(np.exp(epsilon)+len(domain)-1)
    sample_items = np.random.randint(1, len(domain)+1, len(result))
    
    sample_probs = np.random.rand(len(result))
    for i in range(len(result)):
        if sample_probs[i] > p-q:
            # print("change")   
            result[i] = sample_items[i]
    return np.array(result), domain
    
    
def multi_category_finding(file_name, epsilon, fragment_length, k, seed, ground_truth, file_path):
    np.random.seed(seed)
    data = np.load(file_name)
    data = np.random.permutation(data)
    item_domain = max(data[:, 0])
    label_domain = max(data[:, 1])
    max_string_length = len(bin(item_domain)[2:])

    users_group = np.linspace(0, len(data), label_domain+1, dtype=np.int32)
    result = {i:[] for i in range(1, label_domain+1)}
    for i in range(label_domain):
        users_data = data[users_group[i]:users_group[i+1]]
        candidates = find_top_k_per_category(epsilon, users_data, max_string_length, fragment_length, k, seed, i+1)
        result[i+1] = candidates
    
    f1_result = f1_score(result, ground_truth)
    NCG_result = NCG_score(ground_truth, result, top_k_value)
    file = open(file_path+'/initial_result/'+str(epsilon)+'_'+str(top_k_value)+'_result_'+str(seed)+'.txt', 'w')
    file.write(str(f1_result)+'\n')
    file.write(str(NCG_result)+'\n')
    file.close()
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

    file_path = '../result_'+task+'HEC/'
    
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
    args_list = [(file_name, epsilon, fragment_length, top_k_value, seed, ground_truth, file_path) for seed in range(iter_nums)]
    f1 = 0
    ncg = 0
    for i in range(iter_nums):
        result = multi_category_finding(file_name, epsilon, fragment_length, top_k_value, i, ground_truth, file_path)
        f1 += result[0]
        ncg += result[1]
        gc.collect()
    file = open(file_path+str(epsilon)+'_'+str(top_k_value)+'_result.txt', 'w')
    file.write(str(f1/iter_nums)+'\n')
    file.write(str(ncg/iter_nums)+'\n')
    file.close()
    