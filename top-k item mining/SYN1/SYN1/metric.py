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


def ground_truth_obtain(data, k):
    counter_by_group = {}
    for item, group_id in data:
        counter_by_group.setdefault(group_id, Counter())[item] += 1
        
    result = {group_id: [item for item, _ in counter.most_common(k)] for group_id, counter in counter_by_group.items()}
    return result


def NCG_score(sorted_result, estimated_result, k):
    score = 0
    for key in sorted_result.keys():
        for i in range(k):
            if sorted_result[key][i] in estimated_result[key]:
                score += k-i
    return score/len(sorted_result.keys())/(k*(k+1)/2)

    