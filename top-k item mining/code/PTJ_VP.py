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

def OUE_invalid_perturb_old(epsilon, data, user_amount):
    np.random.seed(0)
    p = 1/2
    q = 1/(np.exp(epsilon) + 1)
    bucket_length = len(data)-1
    
    item_count = np.zeros(bucket_length, dtype=np.uint32)
    sample_probs = np.random.rand(user_amount-data[-1], bucket_length+1)
    user_index = 0
    for bucket_index in range(bucket_length):
        for _ in range(data[bucket_index]):
            if sample_probs[user_index][-1]<q:
                user_index += 1
                continue
            for item_ in range(bucket_length):
                if item_ == bucket_index:
                    if sample_probs[user_index][item_] < p:
                        item_count[item_] += 1
                else:
                    if sample_probs[user_index][item_] < q:
                        item_count[item_] += 1
            user_index += 1

    sample_probs = np.random.rand(data[-1], bucket_length+1)
    for elem_index in range(data[-1]):
        if sample_probs[elem_index][-1]<p:
            continue
        for item_ in range(bucket_length):
            if sample_probs[elem_index][item_] < q:
                item_count[item_] += 1
    return item_count


def OUE_initial_perturb(epsilon, data, user_amount):
    bucket_length = len(data)-1
    
    item_count = np.zeros(bucket_length, dtype=np.uint32)
    sample_probs = np.random.rand(user_amount, bucket_length+1)
    user_index = -1
    args_list = []
    for bucket_index in range(bucket_length+1):
        for _ in range(data[bucket_index]):
            user_index += 1
            args_list.append((epsilon, bucket_length, bucket_index, sample_probs[user_index]))
    
    with multiprocessing.Pool(processes=30) as pool:
        result = pool.starmap(OUE_perturb, args_list)
        for count in result:
            item_count += count
    return item_count


def OUE_perturb(epsilon, bucket_length, item_index, probs):
    p = 1/2
    q = 1/(np.exp(epsilon) + 1)
    vectors = np.zeros(bucket_length, dtype=np.uint32)
    if (item_index == bucket_length and probs[-1] < p) or (item_index != bucket_length and probs[-1] < q):
        return vectors  
    for i in range(bucket_length):
        if (item_index == i and probs[i] < p) or (item_index != i and probs[i] < q):
            vectors[i] += 1
    return vectors
    

def suffix_match(data, candidates, suffix_length):
    args_list = [(candidates, data[user_index], suffix_length) for user_index in range(len(data))]
    with multiprocessing.Pool(processes=30) as pool:
            uploading = pool.starmap(suffix_match_function, args_list)
    can_count = np.zeros(len(candidates)+1, dtype=np.uint32)
    for elem, find in uploading:
        if find:
            can_count[elem] += 1
        else:
            can_count[-1] += 1
    return can_count, candidates


def suffix_match_function(candidates, user_data, suffix_length):
    mask = (1 << suffix_length) - 1
    find = False
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
            new_candidates.append(int(padding+bin_candidates[j], 2))
    return    new_candidates 


def find_top_k_per_category(epsilon, data, max_string_length, fragment_length, k):
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
        real_stat, _= suffix_match(users_data, candidates, suffix_length)
        perturbed_est = OUE_initial_perturb(epsilon, real_stat, len(users_data))

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
    
    
def multi_category_finding(file_name, epsilon, fragment_length, k, seed, ground_truth, file_path):
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

    max_string_length = len(bin(count)[2:])
    find_domain = label_domain*k
   
    candidates = find_top_k_per_category(epsilon, data, max_string_length, fragment_length, find_domain)
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
    
    f1_result = f1_score(result, ground_truth)
    NCG_result = NCG_score(ground_truth, result, top_k_value)
    file = open(file_path+'/initial_result/'+str(epsilon)+'_'+str(top_k_value)+'_result_'+str(seed)+'.txt', 'w')
    file.write(str(f1_result)+'\n')
    file.write(str(NCG_result)+'\n')
    file.close()
    print(f1_result, NCG_result)
    return (f1_result, NCG_result)


if __name__ == '__main__':
    # file_name = '../JD_dataset/JD_data_age_sampled.npy'
    # task = 'JD/'
    file_name = '../Anime_dataset/anime_data.npy'
    task = 'anime/'

    file_path = '../result_'+task+'PTJ_VP/'
    
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
    

