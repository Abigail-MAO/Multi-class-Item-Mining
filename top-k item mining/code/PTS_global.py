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


def OUE_initial_perturb(epsilon, data, user_amount, core_number):
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


def suffix_match(data, candidates, suffix_length, core_number):
    mask = (1 << suffix_length) - 1
    buckets = np.zeros(len(candidates)+1, dtype=np.uint32)
    args_list = [(user, candidates, mask) for user in data]
    chunk_size = int(math.ceil(len(data)/core_number))
    with multiprocessing.Pool(processes=core_number) as pool:
        result = pool.starmap(suffix_match_function, args_list, chunksize=chunk_size)
    for i in result:
        buckets[i] += 1
    return buckets

def suffix_match_function(data, candidates, mask):
    for i in range(len(candidates)):
        if data & mask == candidates[i]:
            return i
    return -1


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
            new_candidates.append(int(padding+bin_candidates[j], 2))
    return    new_candidates 


def find_top_k_per_category(epsilon, data, max_string_length, fragment_length, k, candidates, ground, core_number):
    start_length = int(math.log2(k))+1
    num_iterations = 1+int(math.ceil((max_string_length-start_length)/fragment_length))
    data = np.random.permutation(data)
    groups = np.linspace(0, len(data), num_iterations+1, dtype=np.int32)
    first_iterations = num_iterations//2
    for i in range(first_iterations, num_iterations):
        previous_length = start_length+(i-1)*fragment_length
        if previous_length<=1 or i==0:
            previous_length = 1
            candidates = [0, 1]
        suffix_length = start_length + i*fragment_length
        candidates = generate_m_bits_candidates(candidates, previous_length, suffix_length)
        
        users_data = data[groups[i]:groups[i+1]]
        real_stat = suffix_match(users_data, candidates, suffix_length, core_number)
        perturbed_est = OUE_initial_perturb(epsilon, real_stat, len(users_data), core_number)
        indicies = np.argsort(perturbed_est)[::-1][:k]
        candidates = [candidates[i] for i in indicies]

        mask = (1 << suffix_length) - 1
        count = 0
        for elem in ground:
            if elem & mask in candidates:
                count += 1
    return candidates


def GRR(epsilon, data):
    result = copy.deepcopy(data)
    domain = [i for i in range(min(result), max(result)+1)]
    p = np.exp(epsilon)/(np.exp(epsilon)+len(domain)-1)
    q = 1/(np.exp(epsilon)+len(domain)-1)
    sample_items = np.random.randint(1, len(domain)+1, len(result))
    
    sample_probs = np.random.rand(len(result))
    for i in range(len(result)):
        if sample_probs[i] > p-q:
            result[i] = sample_items[i]
    return np.array(result), domain
    
    
def multi_category_finding(file_name, epsilon, fragment_length, k, seed, ground_truth, file_path, core_number):
    np.random.seed(seed)
    data = np.load(file_name)
    max_string_length = len(bin(max(data[:, 0]))[2:])
    data[:, 1], category_info = GRR(epsilon/2, data[:, 1])

    start_length = int(math.log2(k))+1
    num_iterations = 1+int(math.ceil((max_string_length-start_length)/fragment_length))
    quatar = 2
    first_iter_num = num_iterations//quatar
    users_indicies = np.random.choice(len(data), int(len(data)/5), replace=False)
    users_data = data[users_indicies][:, 0]
    users_group = np.linspace(0, len(users_data), first_iter_num+1, dtype=np.int32)
    for i in range(first_iter_num):
        previous_length = start_length+(i-1)*fragment_length
        if previous_length<=1 or i==0:
            previous_length = 1
            candidates = [0, 1]
        suffix_length = start_length + i*fragment_length
        candidates = generate_m_bits_candidates(candidates, previous_length, suffix_length)
        
        user_data = users_data[users_group[i]:users_group[i+1]]
        real_stat = suffix_match(user_data, candidates, suffix_length, core_number)
        '''
        the labels in this part are not utlized in the following part, so the item can be perturbed using the whole privacy budget
        '''
        perturbed_est = OUE_initial_perturb(epsilon, real_stat, len(user_data), core_number)
        indicies = np.argsort(perturbed_est)[::-1][:k*len(category_info)]
        candidates = [candidates[i] for i in indicies]
    data = np.delete(data, users_indicies, axis=0)
    result = {}
    for category in category_info:
        category_data = data[data[:, 1]==category][:, 0]
        class_candidates = find_top_k_per_category(epsilon/2, category_data, max_string_length, fragment_length, k, candidates, ground_truth[category], core_number)
        result[category] = class_candidates
    
    f1_result = f1_score(result, ground_truth)
    NCG_result = NCG_score(ground_truth, result, top_k_value)
    file = open(file_path+'/initial_result/'+str(epsilon)+'_'+str(top_k_value)+'_result_'+str(seed)+'.txt', 'w')
    file.write(str(f1_result)+'\n')
    file.write(str(NCG_result)+'\n')
    file.close()
    return (f1_result, NCG_result)


if __name__ == '__main__':
    # file_name = '../JD_dataset/JD_data_age_sampled.npy'
    # task = 'JD/'
    file_name = '../Anime_dataset/anime_data.npy'
    task = 'anime/'
    file_path = '../result_'+task+'/'+'PTS_global/'
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
    total_f1_score = 0
    total_ncg_score = 0
    iter_count = 0
    for iter_ in range(iter_nums):
        result = multi_category_finding(file_name, epsilon, fragment_length, top_k_value, iter_, ground_truth, file_path, core_number)
        if not result[0]:
            continue
        gc.collect()
        total_f1_score += result[0]
        total_ncg_score += result[1]
        iter_count += 1
    
    file = open(file_path+str(epsilon)+'_'+str(top_k_value)+'_result.txt', 'w')
    file.write(str(total_f1_score/iter_count)+'\n')
    file.write(str(total_ncg_score/iter_count)+'\n')
    file.close()
    