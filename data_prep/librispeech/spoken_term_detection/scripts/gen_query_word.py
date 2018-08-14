#!/usr/bin/env python3
import subprocess
import time
import operator
import os
import random
import pdb




query_wrd_file = 'working_dir/query.text'
test_wrd_file = 'working_dir/test.text'
full_train_ctm_file = 'working_dir/train.ctm.full'
#frame_rate = float(os.environ.get("frame_rate"))
#query_wav_arxiv_root = os.environ.get("query_wav_arxiv")

def gen_wrd_set_and_tf(text_path):
    wrd_set = set()
    wrd_freq = {}
    uttid_set = set()
    with open(text_path, 'r') as f:
        for line in f.readlines():
            uttid_set.add(line.rstrip().split()[0])
            for wrd in line.rstrip().split()[1:]:
                if wrd in wrd_set: wrd_freq[wrd] += 1
                else:
                    wrd_set.add(wrd)
                    wrd_freq[wrd] = 1
    sorted_tf = sorted(wrd_freq.items(), key = operator.itemgetter(1),\
                        reverse=True)
    return sorted_tf, wrd_set, uttid_set


def gen_potential_query_duration(ctm_file, query_uttid_set):
    wrd_dur_map = {}
    with open(ctm_file) as f:
        for line in f.readlines():
            tokens = line.rstrip().split()
            uttid = tokens[0]
            if uttid not in query_uttid_set:
                continue
            wrd = tokens[4]
            duration = float(tokens[3])
            if wrd not in wrd_dur_map.keys():
                wrd_dur_map[wrd] = []
                wrd_dur_map[wrd].append(duration)
            else:
                wrd_dur_map[wrd].append(duration)
    return wrd_dur_map


if __name__ == '__main__':

    timer1 = time.time()
    query_tf, query_wrd_set, query_uttid_set = gen_wrd_set_and_tf(query_wrd_file)
    test_tf, test_wrd_set, _ = gen_wrd_set_and_tf(test_wrd_file)
    potential_query_set = query_wrd_set.intersection(test_wrd_set)
    wrd_dur_map = gen_potential_query_duration(full_train_ctm_file, query_uttid_set)


    selected_set = set()
    for pair in query_tf:
        if pair[0] in potential_query_set and pair[0] in wrd_dur_map.keys():
            wrd_dur_map[pair[0]].sort()
            tmp_list = wrd_dur_map[pair[0]]
            if tmp_list[int(len(tmp_list) / 2)] >= 0.5 and len(pair[0]) > 5 and len(tmp_list) > 10 and len(tmp_list) <= 50:
                selected_set.add(pair[0])
    for query_tf_pair in test_tf:
        if query_tf_pair[0] not in selected_set:  continue
        if query_tf_pair[1] < 10: continue
        print(query_tf_pair[0])


