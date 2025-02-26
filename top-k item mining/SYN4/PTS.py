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
    for bucket_index in range(len(data)-1):
        chunk_size = 100000
        for start in range(0, data[bucket_index], chunk_size):
            end = min(data[bucket_index], start+chunk_size)
            sample_probs = np.random.rand(end-start, bucket_length)
            current_bucket_counts = np.sum(sample_probs[:, bucket_index] < p)
            item_count[bucket_index] += current_bucket_counts
            other_bucket_mask = np.ones((end-start, bucket_length), dtype=bool)
            other_bucket_mask[:, bucket_index] = False
            current_bucket_counts = np.sum((sample_probs < q) & other_bucket_mask, axis=0, dtype=np.uint32)
            item_count += current_bucket_counts
    return item_count


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


def find_top_k_per_category(epsilon, data, max_string_length, fragment_length, k, core_number):
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
        real_stat = suffix_match(users_data, candidates, suffix_length, core_number)
        perturbed_est = OUE_initial_perturb(epsilon, real_stat, len(users_data), core_number)
        # find the top_k candidates to generate the next generation candidates
        indicies = np.argsort(perturbed_est)[::-1][:k]
        candidates = [candidates[i] for i in indicies]
    
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
    
    
def multi_category_finding(data, epsilon, fragment_length, k, seed, ground_truth, file_path, core_number, class_num):
    np.random.seed(seed)
    max_string_length = len(bin(max(data[:, 0]))[2:])
    data[:, 1], category_info = GRR(epsilon/2, data[:, 1])

    result = {}
    for category in category_info:
        category_data = data[data[:, 1]==category][:, 0]
        candidates = find_top_k_per_category(epsilon/2, category_data, max_string_length, fragment_length, k, core_number)
        result[category] = candidates
    
    f1_result = f1_score(result, ground_truth)
    NCG_result = NCG_score(ground_truth, result, top_k_value)
    print(f1_result, NCG_result)
    file = open(file_path+'/initial_result/'+str(class_num)+'_'+str(seed)+'_result.txt', 'w')
    file.write(str(f1_result)+'\n')
    file.write(str(NCG_result)+'\n')
    file.close()
    return (f1_result, NCG_result)


if __name__ == '__main__':
    task = 'SYN2/'

    file_path = '../result_'+task+'/'+'PTS/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        os.makedirs(file_path+'/initial_result')
    
    core_number = 30
    epsilon = 4
    top_k_value = 20
    max_process = core_number
    iter_nums = 20
    fragment_length = 2
    
    class_nums = [i for i in range(10, 55, 10)]
    for class_num in class_nums:
        print(class_num)
        total_f1_score = 0
        total_ncg_score = 0
        iter_count = 0
        
        for iter_ in range(iter_nums):
            print("iteration number", iter_)
            np.random.seed(iter_)
            data = np.load('./data/Class_' + str(class_num) + '.npy', allow_pickle=True)
            data = np.random.permutation(data)
            ground_truth = ground_truth_obtain(data, top_k_value)
            result = multi_category_finding(data, epsilon, fragment_length, top_k_value, iter_, ground_truth, file_path, core_number, class_num)
            if result[0] == 0:
                continue
            total_f1_score += result[0]
            total_ncg_score += result[1]
            iter_count += 1
        print(total_f1_score/iter_count)
        print(total_ncg_score/iter_count)
    
        file = open(file_path+'/'+str(class_num)+'_'+'_result.txt', 'w')
        file.write(str(total_f1_score/iter_count)+'\n')
        file.write(str(total_ncg_score/iter_count)+'\n')
        file.close()
    
    

