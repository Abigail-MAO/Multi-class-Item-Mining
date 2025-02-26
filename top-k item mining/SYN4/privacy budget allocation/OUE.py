import numpy as np 
import multiprocessing


def modified_OUE_v2_old(items, domain_length, epsilon):
    p = 1/2
    q = 1/(np.exp(epsilon) + 1)
    # p = np.exp(epsilon/2)/(np.exp(epsilon/2) + 1)
    # q = 1/(np.exp(epsilon/2) + 1)
    probs = np.random.rand(len(items), domain_length+1)
    vector = [[0 for _ in range(domain_length+1)] for _ in range(len(items))]
    for i in range(len(items)):
        for j in range(domain_length+1):
            if j == items[i]:
                if probs[i][j]<p:
                    vector[i][j] = 1
            else:
                if probs[i][j]<q:
                    vector[i][j] = 1
        if vector[i][-1] == 1:
            for j in range(domain_length+1):
                vector[i][j] = 0
    return np.array(vector)


def modified_OUE_v2(items, domain_length, epsilon):
    p = 1/2
    q = 1/(np.exp(epsilon) + 1)
    # p = np.exp(epsilon/2)/(np.exp(epsilon/2) + 1)
    # q = 1/(np.exp(epsilon/2) + 1)
    probs = np.random.rand(len(items), domain_length+1)
    item_counts = np.zeros(domain_length, dtype=np.int32)
    args_list = []
    for i in range(len(items)):
        args_list.append((items[i], probs[i], domain_length, p, q))

    with multiprocessing.Pool(30) as pool:
        result = pool.starmap(modified_OUE_v2_function, args_list)
    for i in range(len(result)):
        item_counts += result[i][:-1]
    
    return item_counts


def modified_OUE_v2_function(elem, probs, domain_length, p, q):
    vector = np.zeros(domain_length+1, dtype=np.int32)
    for j in range(domain_length+1):
        if j == elem:
            if probs[j]<p:
                vector[j] = 1
        else:
            if probs[j]<q:
                vector[j] = 1
    if vector[-1] == 1:
        for j in range(domain_length+1):
            vector[j] = 0
    return vector


def modified_OUE_v1(items, domain_length, epsilon):
    p = 1/2
    q = 1/(np.exp(epsilon) + 1)
    # p = np.exp(epsilon/2)/(np.exp(epsilon/2) + 1)
    # q = 1/(np.exp(epsilon/2) + 1)
    
    probs = np.random.rand(len(items), domain_length)
    vector = [[0 for _ in range(domain_length)] for _ in range(len(items))]
    for i in range(len(items)):
        for j in range(domain_length):
            if j == items[i]:
                if probs[i][j]<p:
                    vector[i][j] = 1
            else:
                if probs[i][j]<q:
                    vector[i][j] = 1
    return np.array(vector)

def initial_OUE_old(items, domain_length, epsilon):
    p = 1/2
    q = 1/(np.exp(epsilon) + 1)
    # p = np.exp(epsilon/2)/(np.exp(epsilon/2) + 1)
    # q = 1/(np.exp(epsilon/2) + 1)
    # print(p, q)
    
    probs = np.random.rand(len(items), domain_length)
    vector = [[0 for _ in range(domain_length)] for _ in range(len(items))]
    for i in range(len(items)):
        for j in range(domain_length):
            if j == items[i]:
                if probs[i][j]<p:
                    vector[i][j] = 1
            else:
                if probs[i][j]<q:
                    vector[i][j] = 1
    return np.array(vector)


def initial_OUE(items, domain_length, epsilon, core_number):
    p = 1/2
    q = 1/(np.exp(epsilon) + 1)
    # p = np.exp(epsilon/2)/(np.exp(epsilon/2) + 1)
    # q = 1/(np.exp(epsilon/2) + 1)
    
    probs = np.random.rand(len(items), domain_length)
    item_count = np.zeros(domain_length, dtype=np.int32)
    args_list = []
    for i in range(len(items)):
        args_list.append((items[i], probs[i], domain_length, p, q))
    with multiprocessing.Pool(core_number) as pool:
        result = pool.starmap(initial_OUE_function, args_list)
    for i in range(len(result)):
        item_count += result[i]
    return item_count
    

def initial_OUE_function(elem, probs, domain_length, p, q):
    vector = np.zeros(domain_length, dtype=np.int32)
    for j in range(domain_length):
        if j == elem:
            if probs[j]<p:
                vector[j] = 1
        else:
            if probs[j]<q:
                vector[j] = 1
    return vector
