#!/usr/bin/env python3
from sklearn.metrics import average_precision_score
from subprocess import check_output
import random
import pdb


test_table_path = 'test.table'
q_dir = 'test_query_mfc'
d_dir = 'test_doc_mfc'

q_set = set()
d_set = set()
q_d_label = {}
with open(test_table_path) as f:
    for line in f.readlines():
        line = line.rstrip()
        q, d, label = line.split()
        q_set.add(q)
        d_set.add(d)
        q_d_label[(q,d)] = float(label)

average_precision_list = []
counter = 0
for q in q_set:
    labels = []
    dtw_scores = []
    for d in d_set:
        dtw_score = random.random()
        dtw_scores.append(dtw_score)
        labels.append(q_d_label[(q,d)])
    ap = average_precision_score(labels, dtw_scores)
    average_precision_list.append(ap)
    counter += 1
    if counter % 10 == 0:
        print('Progress: {}/{}'.format(counter, len(q_set)))

print('MAP: ' + str(sum(average_precision_list) / len(average_precision_list)))
