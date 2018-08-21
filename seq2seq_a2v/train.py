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





def gen_std_embed_pkl(model, sess, batch_size,
                      iterator, data, output_pkl_name, embed_num_list=None):
    sess.run(iterator.initializer)
    embedding = []
    re_loss_list = []
    utt_num = []
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

        [rnn_code, re_loss] = model.get_tensor_val(
            ['std_rnn_code', 'std_seq2seq_re_loss'], 
            input_X, input_X, utt_mask)

        rnn_code = rnn_code[:len(X), :]
        re_loss_list.append(re_loss)
        utt_num.append(len(X))

        if init == True:
            embed_arr = rnn_code
            init = False
        else:
            embed_arr = np.concatenate([embed_arr, rnn_code] ,axis=0)

    if embed_num_list == None:
        embed_num_list = [1] * len(embed_arr)

    idx = 0
    assert(sum(embed_num_list) == embed_arr.shape[0])
    for num in embed_num_list:
        embedding.append(embed_arr[idx:idx + num])
        idx += num
    with open(output_pkl_name, 'wb') as fp:
        pickle.dump(embedding, fp)

    re_loss = 0.
    for i in range(len(re_loss_list)):
        re_loss += re_loss_list[i] * utt_num[i] / sum(utt_num)
    return re_loss



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


    batch_tr_data, tr_iterator = read_tfrecords(
      librispeech_feat_loc, tr_batch, 'train', shuffle=True)
    batch_dev_data, dev_iterator = read_tfrecords(
      librispeech_feat_loc, tr_batch, 'dev')

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

    #Iinitialize all vars.
    model.init_vars()
    sess.run(tr_iterator.initializer)
    sess.run(dev_iterator.initializer)

    output_msg = ''
    print('=============================================================')
    print('                      Start Training                         ')
    print('=============================================================')
    sys.stdout.flush()
    step = 0
    policy_step = 0

    #max_epoch = int(30000 / (9500 / tr_batch ))
    for epoch in range(1, max_epoch + 1):
        print('[ Epoch {} ]'.format(epoch))
        sys.stdout.flush()

        sess.run(tr_iterator.initializer)
        while True:
            try:
                X, _ = sess.run(batch_tr_data)
            except tf.errors.OutOfRangeError:
                break
            '''
            if len(X) < tr_batch:
                break
            '''

            #utt_mask = np.ones((len(X)))
            # Filter out too short len wrd
            utt_mask = np.sign(np.maximum(40, np.sum(np.sign(np.max(np.abs(X), axis=2)), axis=1)) - 40)
            #utt_mask += 0.1
            noise = np.random.rand(*(X.shape))
            noised_X = X * np.ceil( noise - noise_prob)
            model.train_seq2seq(noised_X, X, utt_mask)
            step += 1


            train_writer.add_summary(
              model.tensorboard_summary(X, X, utt_mask), step)


        if epoch % 5 == 0:
            model.save_vars(model_loc, epoch)

            # STD eval
            std_dir = 'iter_{}_std'.format(epoch)
            subprocess.call('mkdir -p {}'.format(std_dir), shell=True)
            # Generate rnn code for query
            embed_num_list = []
            with open(librispeech_feat_loc + '/query.dev.embed_num') as f:
                for line in f.readlines():
                    embed_num_list.append(int(line.rstrip()))
            re_loss = gen_std_embed_pkl(model, sess, std_batch, 
                dev_query_iterator,
                batch_dev_query, '{}/query.pkl'.format(std_dir), 
                embed_num_list)
            print('reconstruction loss on query: {:.4f}'.format(re_loss))

            embed_num_list = []
            with open(librispeech_feat_loc + '/test.embed_num') as f:
                for line in f.readlines():
                    embed_num_list.append(int(line.rstrip()))

            re_loss = gen_std_embed_pkl(model, sess, std_batch, test_iterator,
                batch_test_data, '{}/doc.pkl'.format(std_dir), embed_num_list)
            print('reconstruction loss on doc: {:.4f}'.format(re_loss))
            subprocess.Popen('utils/std_dev_eval.py {} {}'.format(std_dir, librispeech_std_dev_ans), shell=True)




    print('Done training, save the latest model...')
    model.save_vars(model_loc, epoch)
