#!/usr/bin/env python3

'''
Split train.ctm.full to query.ctm and train.ctm
'''

query_uttid_set = set()
with open('working_dir/query.text') as f:
    for line in f.readlines():
        query_uttid_set.add(line.rstrip().split()[0])

query_ctm_fp = open('working_dir/query.ctm', 'w')
train_ctm_fp = open('working_dir/train.ctm', 'w')

with open('working_dir/train.ctm.full') as f:
    for line in f.readlines():
        if line.rstrip().split()[0] in query_uttid_set:
            query_ctm_fp.write(line)
        else:
            train_ctm_fp.write(line)

query_ctm_fp.close()
train_ctm_fp.close()
