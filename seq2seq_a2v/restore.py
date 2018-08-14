#!/usr/bin/env python3
'''
Training RNN autoencoder
'''
import tensorflow as tf
import numpy as np
import random
import sys
import configparser
from utils.nn_model import SegSeq2SeqAutoEncoder
from utils.vae_nn_model import SegSeq2SeqVAE
from utils.eval import r_val_eval, tolerance_precision, tolerance_recall
import pickle

from sklearn.metrics import average_precision_score
from utils.data_parser import read_table, get_utt_num
import os

import pdb
from tensorflow.python import debug as tf_debug
from utils.data_parser import read_tfrecords
import math
import pickle
import subprocess



def mse(x, y):
    mask = np.max(np.sign(np.abs(x)), axis=2)
    result = np.sqrt(np.power((x - y), 2))
    result = np.mean(result, axis=2)
    result = np.sum(np.sqrt(result) * mask, axis=1) / np.sum(mask, axis=1)
    result = np.mean(result)
    return result



def gen_std_embed_pkl(model, sess, batch_size,
                      iterator, data, output_pkl_name, embed_num_list=None):
    sess.run(iterator.initializer)
    embedding = []
    init = True
    while True:
        try:
            X, _, = sess.run(data)
        except tf.errors.OutOfRangeError:
            break

        if len(X) < batch_size:
            paddings = np.ones(
                (batch_size - len(X), X.shape[1], feature_dim), dtype=float)
            input_X = np.concatenate((X, paddings), axis=0)
            utt_mask = np.ones((len(X)))
            utt_mask = np.concatenate([utt_mask, np.zeros((len(paddings)))])
        else:
            input_X = X
            utt_mask = np.ones((len(X)))

        [rnn_code] = model.get_tensor_val(
            ['std_rnn_code'], input_X, input_X, utt_mask)

        rnn_code = rnn_code[:len(X), :]

        if init == True:
            embed_arr = rnn_code
            init = False
        else:
            embed_arr = np.concatenate([embed_arr, rnn_code] ,axis=0)

    if embed_num_list == None:
        embed_num_list = [1] * len(embed_arr)

    idx = 0
    for num in embed_num_list:
        embedding.append(embed_arr[idx:idx + num])
        idx += num
    with open(output_pkl_name, 'wb') as fp:
        pickle.dump(embedding, fp)



if __name__ == '__main__':

    config = configparser.RawConfigParser()
    config.read('config.cfg')

    #Read config parameters.
    tr_batch = config.getint('train','batch_size')
    tolerance_window = config.getint('seg_eval', 'tolerance_window')
    model_loc = config.get('data', 'model_loc')
    max_epoch = config.getint('train', 'max_epoch')
    noise_prob = config.getfloat('train', 'noise_prob')
    feature_dim = config.getint('data', 'feature_dim')
    dev_utt_num = config.getint('data','dev_utt_num')
    librispeech_feat_loc = config.get('data', 'librispeech_feat_loc')
    librispeech_std_dev_ans = config.get('data', 'librispeech_dev_query_ans')

    #STD eval
    std_batch = config.getint('std_eval', 'batch_size')


    print('=============================================================')
    print('                      Set up models                          ')
    print('=============================================================')
    tf.reset_default_graph()
    sys.stdout.flush()

    sess = tf.Session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)



    # Set up model.
    model = SegSeq2SeqAutoEncoder(config, sess)

    # Set up training operations.
    model.setup_train()
    # Set up tensorboard summary
    model.setup_summary()
    # Write the graph and data into tensorboard.
    dir_name = 'downsampled'
    train_writer = tf.summary.FileWriter('tensorboard/'+ dir_name, sess.graph)

    # Restore all vars.
    #model.restore_vars(model_loc + '-' + str(max_epoch))
    model.restore_vars(model_loc + '-' + str(5))

    with open('/home/allyoushawn/Documents/journal/debug/w1_feats.pkl', 'rb') as fp:
        wrd1_feats = pickle.load(fp)
    with open('/home/allyoushawn/Documents/journal/debug/w2_feats.pkl', 'rb') as fp:
        wrd2_feats = pickle.load(fp)
    with open('/home/allyoushawn/Documents/journal/debug/w3_feats.pkl', 'rb') as fp:
        wrd3_feats = pickle.load(fp)

    data = np.concatenate([wrd1_feats, wrd2_feats, wrd3_feats], axis=0)

    [code] = model.get_tensor_val(['rnn_code'] ,data, data, np.ones((100)))
    with open('code.pkl', 'wb') as fp:
        pickle.dump(code, fp)

