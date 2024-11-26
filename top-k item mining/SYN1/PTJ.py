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
import gc


def OUE_initial_perturb(epsilon, data, user_amount):
    core_number = 30
    p = 1/2
    q = 1/(np.exp(epsilon) + 1)
    sample_invalid = np.random.randint(0, len(data), data[-1])
    for invalid_item in sample_invalid:
        data[invalid_item] += 1
    bucket_length = len(data)-1

    item_count = np.zeros(bucket_length, dtype=np.uint32)
    print(len(data), user_amount)
    for bucket_index in range(len(data)-1):
        chunck_size = 100000
        for start in range(0, data[bucket_index], chunck_size):
            end = min(start+chunck_size, data[bucket_index])
            sample_probs = np.random.rand(end-start, bucket_length)
            current_bucket_counts = np.sum(sample_probs[:, bucket_index] < p)
            item_count[bucket_index] += current_bucket_counts
            other_buckets_mask = np.ones((end - start, bucket_length), dtype=bool)
            other_buckets_mask[:, bucket_index] = False  
            other_buckets_counts = np.sum((sample_probs < q) & other_buckets_mask, axis=0, dtype=np.uint32)
            item_count += other_buckets_counts
    return item_count

    
# match the suffix of the data, if not match then record them in the last bucket
def suffix_match_old(data, candidates, suffix_length):
    mask = (1 << suffix_length) - 1
    buckets = np.zeros(len(candidates)+1, dtype=np.uint32)
    for user in data:
        find = False
        for i in range(len(candidates)):
            if user & mask == candidates[i]:
                buckets[i] += 1
                find = True
                break
        if not find:
            buckets[-1] += 1
    return buckets


def suffix_match(data, candidates, suffix_length):
    core_number = 30
    mask = (1 << suffix_length) - 1
    buckets = np.zeros(len(candidates)+1, dtype=np.uint32)
    args_list = [(data[i], candidates, mask) for i in range(len(data))]
    with multiprocessing.Pool(processes=core_number) as pool:
        result = pool.starmap(suffix_match_function, args_list)
    for i in range(len(result)):
        buckets[result[i]] += 1
    return buckets

def suffix_match_function(data, candidates, mask):
    for i in range(len(candidates)):
        if data& mask == candidates[i]:
            return i
    return -1


def generate_m_bits_candidates(candidates, start_bit, end_bit, domain_size):
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


def find_top_k_per_category(epsilon, data, max_string_length, fragment_length, k, domain_size):
    start_length = int(math.log2(k))+1
    num_iterations = 1+int(math.ceil((max_string_length-start_length)/fragment_length))
    data = np.random.permutation(data)
    groups = np.linspace(0, len(data), num_iterations+1, dtype=np.int32)
    
    for i in range(num_iterations):
        print(i, num_iterations, "iteration_num")
        previous_length = start_length+(i-1)*fragment_length
        if previous_length<=1 or i==0:
            previous_length = 1
            candidates = [0, 1]
        suffix_length = start_length + i*fragment_length
        candidates = generate_m_bits_candidates(candidates, previous_length, suffix_length, domain_size)
        
        users_data = data[groups[i]:groups[i+1]]
        real_stat = suffix_match(users_data, candidates, suffix_length)
      
        perturbed_est = OUE_initial_perturb(epsilon, real_stat, len(users_data))
        gc.collect()
       
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
    
    
def multi_category_finding(initial_data, epsilon, fragment_length, k, seed, ground_truth, file_path, class_num):
    np.random.seed(seed)
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

    max_string_length = len(bin(count)[2:])
    find_domain = label_domain*k
    
    candidates = find_top_k_per_category(epsilon, data, max_string_length, fragment_length, find_domain, domain_size)
    result = {i:[] for i in range(1, label_domain+1)}
    for elem in candidates:
        if elem not in elem_domain.keys():
            continue
        label, item = elem_domain[elem]
        if len(result[label]) < k:
            result[label].append(item)
    for i in result.keys():
        if len(result[i]) < k:
            result[i].extend([None]*(k-len(result[i])))

    f1_result = f1_score(result, ground_truth)
    NCG_result = NCG_score(ground_truth, result, top_k_value)
    file = open(file_path+'/initial_result/'+str(class_num)+'_result_'+str(seed)+'.txt', 'w')
    file.write(str(f1_result)+'\n')
    file.write(str(NCG_result)+'\n')
    file.close()
    print(f1_result, NCG_result)
    return (f1_result, NCG_result)


if __name__ == '__main__':
    task = 'SYN2'

    file_path = '../result_'+task+'/'+'PTJ/'
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
        total_f1_score = 0
        total_ncg_score = 0
        iter_count = 0
        
        for iter_ in range(iter_nums):
            np.random.seed(iter_)
            data = np.load('./data/Class_' + str(class_num) + '.npy', allow_pickle=True)
            data = np.random.permutation(data)
            ground_truth = ground_truth_obtain(data, top_k_value)
            result = multi_category_finding(data, epsilon, fragment_length, top_k_value, iter_, ground_truth, file_path, class_num)
            if result[0] == 0:
                continue
            total_f1_score += result[0]
            total_ncg_score += result[1]
            iter_count += 1
            gc.collect()
        print(total_f1_score/iter_count)
        print(total_ncg_score/iter_count)
    
        file = open(file_path+'/'+str(class_num)+'_'+'_result.txt', 'w')
        file.write(str(total_f1_score/iter_count)+'\n')
        file.write(str(total_ncg_score/iter_count)+'\n')
        file.close()
    
    
    

