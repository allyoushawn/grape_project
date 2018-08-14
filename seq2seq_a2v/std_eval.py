#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import sys
from utils import nn_model
from utils.sub_sequence_match_model import SubSequenceMatchModel
from utils.data_parser import get_feat_dim, get_utt_num, data_loader
from utils.data_parser import load_all_bounds_through_file_for_std, read_table
import configparser
from sklearn.metrics import average_precision_score
import pickle
import os

import time

import pdb




def print_progress(progress, total,  output_msg):
    sys.stdout.write('\b' * len(output_msg))
    output_msg = 'Progress: {}/{}'.format(progress, total)
    sys.stdout.write(output_msg)
    sys.stdout.flush()
    return output_msg




if __name__ == '__main__':
    config = configparser.RawConfigParser()
    config.read('config.cfg')
    sys.stdout.flush()

    max_len = config.getint('data', 'max_length')
    de_batch = config.getint('std_eval', 'batch_size')
    test_doc_scp = config.get('data', 'test_scp')
    test_bound_file = config.get('data', 'test_bnd_file')
    test_query_scp = config.get('data', 'test_query_scp')
    test_table = config.get('data', 'test_table')
    test_scp = config.get('data', 'std_test_scp')
    corpus_type = config.get('data', 'corpus_type')
    model_loc = config.get('data', 'model_loc')
    feat_loc = config.get('data', 'feat_loc')
    '''
    print('=============================================================')
    print('                      Loading data                          ')
    print('=============================================================')
    '''
    sys.stdout.flush()
    feature_dim = get_feat_dim(test_query_scp)
    #print('feature dim: ' + str(feature_dim))
    config.set('data', 'feature_dim', feature_dim)

    #Load all bounds through bound file for oracle usage.
    all_bounds_dict = load_all_bounds_through_file_for_std(test_bound_file, corpus_type)
    '''
    print('=============================================================')
    print('                      Set up models.                         ')
    print('=============================================================')
    '''

    sess = tf.Session()
    model = nn_model.NeuralNetwork(config, sess)
    model.restore_vars(model_loc + '-' + sys.argv[1])

    #Set up subsequent matching ops
    match_model = SubSequenceMatchModel(sess)

    Y, uttid_list = read_table(test_table, test_query_scp, test_doc_scp, test_scp)

    print('')
    output_msg = ''
    counter = 0
    uttid_list_idx = 0
    scores = []
    test_re_loss = 0.0
    block_utt_counter = 0
    utt_block_idx = 0
    '''
    print('=============================================================')
    print('                      Start Decoding                         ')
    print('=============================================================')
    '''
    std_feat_pickle_name = feat_loc
    std_feat_pickle_name += '/std_table_feat_0'
    std_feat_pickle_name += '.pkl'
    with open(std_feat_pickle_name, 'rb') as fp:
        utterance_block = pickle.load(fp)

    while counter < len(Y):
        if block_utt_counter == len(utterance_block):
            utt_block_idx += 1
            std_feat_pickle_name = feat_loc
            std_feat_pickle_name += 'std_table_feat_{}'.format(utt_block_idx)
            std_feat_pickle_name += '.pkl'
            if os.path.isfile(std_feat_pickle_name) == False:
                raise IOError('{} not found...'.format(std_feat_pickle_name))
            with open(std_feat_pickle_name, 'rb') as fp:
                utterance_block = pickle.load(fp)
                block_utt_counter = 0

        utt_batch_size = min(2 * de_batch,
                             len(utterance_block) - block_utt_counter)
        X =  utterance_block[block_utt_counter: block_utt_counter + utt_batch_size,:,:]
        block_utt_counter += utt_batch_size
        std_pair_num = int(utt_batch_size / 2)
        #batch_size = min(de_batch, len(Y) - counter)
        seg_action = np.zeros(X.shape[:2], dtype=np.int32)
        seg_action_tmp = model.get_greedy_segmentation(X, X, seg_action, len(X))
        seg_action = seg_action_tmp.copy()
        '''
        for i, uttid in enumerate(uttid_list[uttid_list_idx:uttid_list_idx + 2 * batch_size]):
            if uttid not in all_bounds_dict.keys():  continue
            for bnd in all_bounds_dict[uttid]:
                oracle_seg[i][bnd] = 1
        '''
        counter += std_pair_num
        uttid_list_idx += utt_batch_size

        [rnn_code, seq2seq_re_loss] = model.get_tensor_val(['rnn_code', \
                                          'seq2seq_re_loss'], X, X,
                                          seg_action, len(X))
        std_pair_idx = 0
        test_re_loss += seq2seq_re_loss * float(len(X)) / (2 * len(Y))

        while std_pair_idx < utt_batch_size:
            query_embed_arr = rnn_code[std_pair_idx].copy()
            doc_embed_arr = rnn_code[std_pair_idx + 1].copy()
            query_arr = query_embed_arr[~(query_embed_arr==0).all(1)]
            doc_arr = doc_embed_arr[~(doc_embed_arr==0).all(1)]
            sub_match_scores = []
            if doc_arr.shape[0] < query_arr.shape[0]:
                sub_match_scores.append(-1. * max_len)
            else:
                #sub_match_score = match_model.match_score(doc_arr, query_arr)
                doc_arr /= np.tile(np.expand_dims(np.linalg.norm(doc_arr, 
                                                axis=1),1),doc_arr.shape[1])
                query_arr /= np.tile(np.expand_dims(np.linalg.norm(query_arr, 
                                                axis=1),1),query_arr.shape[1])
                for match_idx in range(doc_arr.shape[0] - query_arr.shape[0] + 1):
                    score = query_arr * doc_arr[match_idx:match_idx + query_arr.shape[0],:]
                    score = np.sum(score, axis=1)
                    score = (score + 1) / 2
                    sub_match_scores.append(np.prod(score))

            scores.append(max(sub_match_scores))
            std_pair_idx += 2
        output_msg = print_progress(counter, len(Y) , output_msg)
    print('')
    #MAP eval
    test_arxiv_size = get_utt_num(test_doc_scp)
    label = np.array(Y)
    scores = np.array(scores)
    map_list = []
    idx_ptr = 0
    while idx_ptr < len(Y):
        batch_label = label[idx_ptr:idx_ptr + test_arxiv_size].copy()
        batch_scores = scores[idx_ptr:idx_ptr + test_arxiv_size].copy()
        map_list.append(average_precision_score(batch_label, batch_scores))
        idx_ptr += test_arxiv_size
    map_score = sum(map_list) / len(map_list)
    print('MAP: {:.4f}'.format(map_score))
    print('Reconstruciton loss: {:.4f}'.format(test_re_loss))


