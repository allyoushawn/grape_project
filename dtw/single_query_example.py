#!/usr/bin/env python3
from sklearn.metrics import average_precision_score
import pdb
import subprocess
import sys

if len(sys.argv) != 6:
    print('Usage: singel_query_expmple.py <query_scp> <doc_scp> <query_idx> <ans> <mfc_dir>')
    quit()

query_scp = sys.argv[1]
doc_scp = sys.argv[2]
query_idx = int(sys.argv[3])
ans = sys.argv[4]
mfc_dir = sys.argv[5]

query_uttid_list = []
with open(query_scp, 'r') as f:
    for line in f.readlines():
        uttid = line.split()[0]
        query_uttid_list.append(uttid)

doc_uttid_list = []
with open(doc_scp, 'r') as f:
    for line in f.readlines():
        uttid = line.split()[0]
        doc_uttid_list.append(uttid)

labels = [0] * len(doc_uttid_list)

with open(ans) as fp:
    for idx, line in enumerate(fp.readlines()):
        if idx != query_idx:  continue
        else:
            for ans in [ int(x) for x in line.rstrip().split()]:
                labels[ans] = 1
            break
query_uttid = query_uttid_list[query_idx]
scores = []
for doc_uttid in doc_uttid_list:
    q_path = mfc_dir + '/' + query_uttid + '.mfc'
    d_path = mfc_dir + '/' + doc_uttid + '.mfc'
    dtw_score = float(subprocess.check_output(['./run_dtw', q_path, d_path]))
    scores.append(dtw_score)

'''
k = 10
selected_index = sorted(range(len(scores)), key=lambda k: scores[k])[-1 * k:]
hit = 0
for idx in selected_index:
    if labels[idx] == 1:    hit += 1
print('{:.4f}'.format(hit / k))
'''

print('{:.4f}'.format(average_precision_score(labels, scores)))



