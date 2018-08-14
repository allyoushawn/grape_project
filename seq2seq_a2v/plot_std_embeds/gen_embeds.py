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

model_num = 200



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

    with open(output_pkl_name, 'wb') as fp:
        pickle.dump(embed_arr, fp)



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
    librispeech_std_dev_ans = config.get('data', 'librispeech_dev_query_ans')
    librispeech_feat_loc = '/media/hdd/csie/features/journal/seq2seq_downsampled'

    #STD eval
    std_batch = config.getint('std_eval', 'batch_size')


    print('=============================================================')
    print('                      Set up models                          ')
    print('=============================================================')
    tf.reset_default_graph()
    sys.stdout.flush()

    sess = tf.Session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)


    batch_dev_query, dev_query_iterator = read_tfrecords(
      librispeech_feat_loc, std_batch, 'query.dev')
    batch_test_data, test_iterator = read_tfrecords(
      librispeech_feat_loc, std_batch, 'test')


    #Set up model.
    model = SegSeq2SeqAutoEncoder(config, sess)

    #Set up training operations.
    model.setup_train()
    #Set up tensorboard summary
    model.setup_summary()
    #Write the graph and data into tensorboard.
    dir_name = 'downsampled'
    train_writer = tf.summary.FileWriter('tensorboard/'+ dir_name, sess.graph)

    model.restore_vars(model_loc + '-' + str(model_num))

    sys.stdout.flush()
    step = 0



    # STD eval
    std_dir = 'plot_std'
    subprocess.call('mkdir -p {}'.format(std_dir), shell=True)
    # Generate rnn code for query
    gen_std_embed_pkl(model, sess, std_batch, dev_query_iterator,
        batch_dev_query, '{}/query.pkl'.format(std_dir))

    embed_num_list = []
    with open(librispeech_feat_loc + '/test.embed_num') as f:
        for line in f.readlines():
            embed_num_list.append(int(line.rstrip()))

    gen_std_embed_pkl(model, sess, std_batch, test_iterator,
        batch_test_data, '{}/doc.pkl'.format(std_dir), embed_num_list)

    with open(std_dir + '/doc.pkl', 'rb') as fp:
        doc_embed = pickle.load(fp)
    with open(std_dir + '/query.pkl', 'rb') as fp:
        query_embed = pickle.load(fp)
    with open(std_dir + '/code.pkl', 'wb') as fp:
        pickle.dump(np.concatenate([doc_embed, query_embed], axis=0), fp)



