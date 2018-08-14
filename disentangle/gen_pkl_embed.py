#!/usr/bin/env python3
import pickle
import sys
import numpy as np
import pdb

if len(sys.argv) != 4:
    print('Usage: gen_pkl_embed.py <embed_file> <ali_file> <output_pkl>')
    quit()

embed_file = sys.argv[1]
ali_file = sys.argv[2]
op_pkl = sys.argv[3]


# Read ali to get utterance id order
uttid_list = []
uttid_set = set()
with open(ali_file) as f:
    for line in f.readlines():
        line_uttid = line.rstrip().split()[0]
        if line_uttid not in uttid_set:
            uttid_set.add(line_uttid)
            uttid_list.append(line_uttid)


# A dict map each uttid with an embed list
uttid_embed_dict = {}


uttid = 'init'
with open(embed_file) as f:
    for line in f.readlines():
        tokens = line.rstrip().split()
        line_uttid = tokens[-1]
        if line_uttid != uttid:
            if uttid != 'init':
                uttid_embed_dict[uttid] = embed_list[:]
            uttid = line_uttid
            embed_list = []
        feats = [ float(x) for x in tokens[:-1] ]
        embed_list.append(np.expand_dims(np.array(feats), axis=0))

uttid_embed_dict[uttid] = embed_list[:]
embed_list = []
for uttid in uttid_list:
    if uttid not in uttid_embed_dict.keys(): 
        continue
    embed_list.append(np.concatenate(uttid_embed_dict[uttid], axis=0))
print(len(embed_list))
with open(op_pkl, 'wb') as fp:
    pickle.dump(embed_list, fp)


