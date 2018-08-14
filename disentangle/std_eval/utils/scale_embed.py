#!/usr/bin/env python3
import pickle
from sklearn.metrics import average_precision_score
from sklearn import preprocessing
import numpy as np
import pdb
import sys




def preprocessing_embed(query_arr_list, doc_arr_list):
    total_query = query_arr_list[0]
    for i in range(len(query_arr)):
        if i == 0:    continue
        total_query = np.concatenate([total_query, query_arr_list[i]], axis=0)
    query_embed_num = len(total_query)

    total_doc = doc_arr_list[0]
    for i in range(len(doc_arr_list)):
        if i == 0:    continue
        total_doc = np.concatenate([total_doc, doc_arr_list[i]], axis=0)

    total_embedding = np.concatenate([total_query, total_doc], axis=0)
    #total_embedding = preprocessing.normalize(total_embedding)
    total_embedding = preprocessing.scale(total_embedding)
    total_embedding = preprocessing.normalize(total_embedding)

    scaled_query_embed = total_embedding[:query_embed_num]
    scaled_doc_embed = total_embedding[query_embed_num:]

    scaled_query_arr_list = []
    for i in range(query_embed_num):
        scaled_query_arr_list.append(scaled_query_embed[i:i+1])

    idx = 0
    utt = 0
    scaled_doc_arr_list = []
    while len(scaled_doc_arr_list) < len(doc_arr_list):
        scaled_doc_arr_list.append(scaled_doc_embed[idx:idx + len(doc_arr_list[utt])])
        idx += len(doc_arr_list[utt])
        utt += 1


    return scaled_query_arr_list, scaled_doc_arr_list




if len(sys.argv) != 3:
    print('Usage: scaled.py <query_pkl> <doc_pkl> ')
    quit()
query_pkl = sys.argv[1]
doc_pkl = sys.argv[2]

with open(query_pkl, 'rb') as fp:
    query_arr = pickle.load(fp)

with open(doc_pkl, 'rb') as fp:
    doc_arr_list = pickle.load(fp)

query_arr, doc_arr_list = preprocessing_embed(query_arr, doc_arr_list)

with open(query_pkl+'.scaled', 'wb') as fp:
    pickle.dump(query_arr, fp)

with open(doc_pkl+'.scaled', 'wb') as fp:
    pickle.dump(doc_arr_list, fp)
