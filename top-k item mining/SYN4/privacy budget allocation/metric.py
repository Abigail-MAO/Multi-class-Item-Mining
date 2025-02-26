import scipy.stats as stats
import numpy as np
from collections import Counter

def f1_score(result, ground_truth):
    score = 0
    for catefory_label in result.keys():
        if len(result[catefory_label]) == 0:
            continue
        p = len(set(result[catefory_label]).intersection(set(ground_truth[catefory_label])))/len(result[catefory_label])
        r = len(set(result[catefory_label]).intersection(set(ground_truth[catefory_label])))/len(ground_truth[catefory_label])
        if p+r == 0:
            score += 0
            continue
        score += 2*p*r/(p+r)
    return score/len(result.keys())


def kendalltau_score(result, ground_truth):
    tau_score = 0
    for catefory_label in result.keys():
        tau_score += stats.kendalltau(result[catefory_label], ground_truth[catefory_label])[0]
    return tau_score/len(result.keys())


def ground_truth_obtain(file_name, k):
    data = np.load(file_name, allow_pickle=True)
    counter_by_group = {}
    for item, group_id in data:
        counter_by_group.setdefault(group_id, Counter())[item] += 1
        
    result = {group_id: [item for item, _ in counter.most_common(k)] for group_id, counter in counter_by_group.items()}
    return result


def rank_correction(sorted_result, estimated_result):
    result = {}
    for key in sorted_result.keys():
        item_index = {}
        for i in range(len(sorted_result[key])):
            item_index[sorted_result[key][i]] = i+1
        est_array = np.zeros(len(estimated_result[key]))
        for i in range(len(estimated_result[key])):
            if estimated_result[key][i] not in item_index.keys():
                est_array[i] = len(sorted_result[key])
            else:
                est_array[i] = item_index[estimated_result[key][i]]
        result[key] = est_array
    return result

def NCG_score(sorted_result, estimated_result, k):
    score = 0
    for key in sorted_result.keys():
        for i in range(k):
            if sorted_result[key][i] in estimated_result[key]:
                score += k-i
    return score/len(sorted_result.keys())/(k*(k+1)/2)


if __name__=='__main__':
    result = rank_correction(ground_truth_obtain(10))
    print(result)
    