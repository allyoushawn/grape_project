#!/usr/bin/env python3
import subprocess
import pdb

query_scp = '/home/allyoushawn/features/librispeech_feats/query.dev.39.cmvn.scp'
doc_scp = '/home/allyoushawn/features/librispeech_feats/test.39.cmvn.scp'
ans = '/home/allyoushawn/Documents/data_prep/librispeech/spoken_term_detection/query/query.dev.ans'
mfc_dir = '/home/allyoushawn/features/librispeech_feats/separate_mfcc_dir'


with open(query_scp) as f:
    query_count = len(f.readlines())

subprocess.call('rm -f querywise_result ', shell=True)

op_f = open('jobs', 'w')
for query_idx in range(query_count):
    op_f.write('./single_query_example.py {} {} {} {} {} >>querywise_result\n'.format(query_scp, doc_scp, str(query_idx), ans, mfc_dir))

subprocess.call('cat jobs | parallel --no-notice -j 16 ', shell=True)

scores = []
with open('querywise_result') as f:
    for line in f.readlines():
        scores.append(float(line.rstrip()))

print('MAP: {:.4f}'.format(sum(scores) / len(scores)))


