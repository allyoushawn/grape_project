#!/usr/bin/env python3
import pickle
from sklearn.metrics import average_precision_score
import numpy as np
import pdb
import sys

if len(sys.argv) != 5:
    print('Usage: singel_query_expmple.py <query_pkl> <doc_pkl> <query_idx> <ans>')
    quit()
query_pkl = sys.argv[1]
doc_pkl = sys.argv[2]
query_idx = int(sys.argv[3])
ans = sys.argv[4]

with open(query_pkl, 'rb') as fp:
    query_arr = pickle.load(fp)

query_arr = query_arr[query_idx]
with open(doc_pkl, 'rb') as fp:
    doc_arr_list = pickle.load(fp)

labels = [0] * len(doc_arr_list)


with open(ans) as fp:
    for idx, line in enumerate(fp.readlines()):
        if idx != query_idx:  continue
        else:
            for ans in [ int(x) for x in line.rstrip().split()]:
                labels[ans] = 1

scores = []
for doc_arr in doc_arr_list:
    sub_match_scores = []
    if doc_arr.shape[0] < query_arr.shape[0]:
        doc = query_arr
        query = doc_arr
    else:
        doc = doc_arr
        query = query_arr

    for match_idx in range(doc.shape[0] - query.shape[0] + 1):
        q = query / np.linalg.norm(query)
        d = doc[match_idx:match_idx + query.shape[0],:] / np.linalg.norm(doc[match_idx:match_idx + query.shape[0],:])
        score = q * d

        #score = query * doc[match_idx:match_idx + query.shape[0],:]
        score = np.sum(score, axis=1)
        score = (score + 1) / 2
        sub_match_scores.append(np.prod(score))
    scores.append(max(sub_match_scores))

print('{:.4f} {}'.format(average_precision_score(labels, scores), str(query_idx)))



