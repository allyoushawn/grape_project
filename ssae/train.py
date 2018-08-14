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




def sample_bnds(X, std_mode=False):
    seg_action = model.sample_bnds(X, X, std_mode)
    return seg_action


def detect_bnds(X):
    seg_action = np.zeros(X.shape[:2], dtype=np.int32)
    seg_action_tmp = \
      model.get_greedy_segmentation(X, X, seg_action)
    return seg_action_tmp


def assign_bnds(X, bnds):
    seg_action = np.zeros(X.shape[:2], dtype=np.int32)
    for i, utt_bnds in enumerate(
     bnds[counter:counter + batch_size]):
        for bnd in utt_bnds:
            seg_action[i][bnd] = 1
    return seg_action


def gen_std_embed_pkl(model, sess, batch_size,
                      iterator, data, output_pkl_name):
    sess.run(iterator.initializer)
    embedding = []
    while True:
        try:
            X, _, _ = sess.run(data)
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

        seg_action = sample_bnds(input_X, std_mode=True)
        seq_len_filter = np.ones_like(seg_action)
        [rnn_code] = model.get_tensor_val(
            ['std_rnn_code'], input_X, input_X, seg_action, utt_mask, seq_len_filter)

        rnn_code = rnn_code[:len(X), :, :]

        for i in range(len(X)):
            embed_arr = rnn_code[i].copy()
            embedding.append(embed_arr[~(embed_arr==0).all(1)])

    with open(output_pkl_name, 'wb') as fp:
        pickle.dump(embedding, fp)



def generate_seq_len_fileter(seg_action):
    code_mask = seg_action.copy()
    code_mask[:, -1] = np.ones_like(code_mask[:, -1])
    reverse = np.ones_like(code_mask) - code_mask
    seq_len = np.zeros_like(code_mask)
    seq_len[:, 0] = np.ones_like(seq_len[:, 0])
    for i in range(1, code_mask.shape[1]):
        seq_len[:, i] = np.where(code_mask[:, i-1] == 0, 
            seq_len[:, i-1] + 1, np.ones_like(seq_len[:, i]))
    seq_len *= code_mask

    thresh = 0
    ret = np.floor((np.sign(seq_len - thresh) + 1) / 2)

    ret_mask = np.zeros_like(code_mask)
    ret_mask[:, -1] = ret[:, -1]
    tmp = ret[:, -1]

    for i in reversed(range(seg_action.shape[1] - 1)):
        ret_mask[:, i] = np.where(code_mask[:, i] == 0, 
            ret_mask[:, i+1], ret[:, i])
    return ret_mask



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
    model_type = 'seq2seq'
    if model_type != 'vae':
        model = SegSeq2SeqAutoEncoder(config, sess)
    else:
        model = SegSeq2SeqVAE(config, sess)

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


    for epoch in range(1, max_epoch + 1):
        print('[ Iteration {} ]'.format(epoch))

        print('    Train Seq2Seq')
        for i in range(5):
            try:
                X, _, _ = sess.run(batch_tr_data)
            except tf.errors.OutOfRangeError:
                sess.run(tr_iterator.initializer)
                X, _, _ = sess.run(batch_tr_data)

            if len(X) < tr_batch:
                sess.run(tr_iterator.initializer)
                X, _, _ = sess.run(batch_tr_data)

            utt_mask = np.ones((len(X)))
            batch_detected_bnds_arr = sample_bnds(X)
            seq_len_filter = generate_seq_len_fileter(batch_detected_bnds_arr)
            noise = np.random.rand(*(X.shape))
            noised_X = X * np.ceil( noise - noise_prob)
            model.train_seq2seq(noised_X, X,
             batch_detected_bnds_arr, utt_mask, seq_len_filter)


        #train policy rnn
        print('')
        print('    Train policy RNN')
        sys.stdout.flush()
        try:
            X, _, _ = sess.run(batch_tr_data)
        except tf.errors.OutOfRangeError:
            sess.run(tr_iterator.initializer)
            X, _, _ = sess.run(batch_tr_data)

        if len(X) < tr_batch:
            sess.run(tr_iterator.initializer)
            X, _, _ = sess.run(batch_tr_data)

        utt_mask = np.ones((len(X)))
        batch_detected_bnds_arr = sample_bnds(X)
        seq_len_filter = np.ones_like(batch_detected_bnds_arr)
        batch_reward = model.calculate_reward(X, X, batch_detected_bnds_arr,
                                              utt_mask, seq_len_filter)
        reward_baseline = np.mean(batch_reward) * np.ones_like(batch_reward)
        model.train_policy(X, X, batch_detected_bnds_arr,
                           batch_reward, reward_baseline)

        train_writer.add_summary(
          model.tensorboard_summary(X, X,
            batch_detected_bnds_arr, utt_mask, batch_reward, seq_len_filter), epoch)

        if epoch % 50 != 0:  continue

        #dev eval r-value
        print('')
        precision = 0.0
        recall = 0.0
        dev_re_loss = 0.0
        dev_embed_num_ratio = 0.0
        dev_sample_num = 5
        counter = 0

        sess.run(dev_iterator.initializer)
        total_batch = int(math.ceil(dev_utt_num / tr_batch))
        for _ in range(total_batch):
            X, bnds, _ = sess.run(batch_dev_data)
            if len(X) < tr_batch:
                #paddings = np.zeros((tr_batch - len(X), batch_max_len, feature_dim), dtype=float)
                paddings = np.ones((tr_batch - len(X), X.shape[1], feature_dim), dtype=float)
                input_X = np.concatenate((X, paddings), axis=0)
                utt_mask = np.ones((len(X)))
                utt_mask = np.concatenate([utt_mask, np.zeros((len(paddings)))])
            else:
                input_X = X
                utt_mask = np.ones((len(X)))

            for _ in range(dev_sample_num):
                seg_action = sample_bnds(input_X)
                seq_len_filter = np.ones_like(seg_action)
                if model_type != 'vae':
                    [seq2seq_re_loss, rnn_code]= model.get_tensor_val(
                      ['seq2seq_re_loss', 'rnn_code'],
                      input_X, input_X, seg_action, utt_mask, seq_len_filter)
                else:
                    [seq2seq_re_loss, rnn_code]= model.get_tensor_val(
                      ['lower_bound', 'rnn_code'],
                      input_X, input_X, seg_action, utt_mask, seq_len_filter)
                rnn_code = rnn_code[:len(X), :, :]
                embed_idx = np.sign(np.max(np.abs(rnn_code), axis=2))
                mask = np.sign(np.max(np.abs(X), axis=2))

                batch_dev_embed_num_ratio = \
                 np.sum(embed_idx, axis=1) / np.sum(mask, axis=1)
                for utt_idx in range(len(X)):
                    #calculate tolerance precision and recall.
                    seg_bnds = np.nonzero(embed_idx[utt_idx])[0].tolist()

                    #remove the last one since it is the end of the utterance
                    seg_bnds.sort()
                    seg_bnds = seg_bnds[:-1]
                    bnds_list = np.nonzero(bnds[utt_idx])[0].tolist()
                    if len(seg_bnds) == 0: continue
                    precision += (1. / (dev_utt_num * dev_sample_num)) * \
                        tolerance_precision(bnds_list,
                                            seg_bnds, tolerance_window)
                    recall += (1. / (dev_utt_num * dev_sample_num)) * \
                        tolerance_recall(bnds_list,
                                         seg_bnds, tolerance_window)
                dev_re_loss += seq2seq_re_loss * float(len(X)) / (dev_utt_num * dev_sample_num)
                dev_embed_num_ratio += \
                 np.mean(batch_dev_embed_num_ratio) * float(len(X)) / (dev_utt_num * dev_sample_num)
        if model_type != 'vae':
            print('Dev. seq2seq_re_loss: {:.4f}'.format(dev_re_loss))
        if model_type == 'vae':
            print('Dev. ELBO: {:.4f}'.format(dev_re_loss))
        print('Dev. embed_num_ratio: {:.4f}'.format(dev_embed_num_ratio))

        print('')
        title =  'Precision Recall F1-score R-value'
        print(title)
        recall *= 100
        precision *= 100
        if recall == 0. or precision == 0.:
            f_score = -1.
            r_val = 0.
        else:
            f_score = (2 * precision * recall) / (precision + recall)
            r_val = r_val_eval(precision, recall)
        print('Seg. Performance: {:.4f} {:.4f} {:.4f} {:.4f}'. \
            format(precision, recall, f_score, r_val))
        print('')
        sys.stdout.flush()


        # STD eval
        std_dir = 'iter_{}_std'.format(epoch)
        subprocess.call('mkdir -p {}'.format(std_dir), shell=True)
        # Generate rnn code for query
        gen_std_embed_pkl(model, sess, std_batch, dev_query_iterator,
            batch_dev_query, '{}/query.pkl'.format(std_dir))

        gen_std_embed_pkl(model, sess, std_batch, test_iterator,
            batch_test_data, '{}/doc.pkl'.format(std_dir))
        subprocess.Popen('utils/std_dev_eval.py {} {}'.format(std_dir, librispeech_std_dev_ans), shell=True)

        if epoch % 500 == 0:
            model.save_vars(model_loc, epoch)



    print('Done training, save the latest model...')
    model.save_vars(model_loc, epoch)
