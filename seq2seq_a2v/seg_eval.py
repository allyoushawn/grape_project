#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import sys
from utils import nn_model
from utils.data_parser import bound_loader, load_data_into_mem
from utils.data_parser import padding, get_specific_utt_bound
from utils.data_parser import load_all_bounds_through_file
from utils.eval import r_val_eval, tolerance_precision, tolerance_recall
import configparser
import pdb

#Data config
enable_plt = False

if enable_plt == True:
    import matplotlib.pyplot as plt


def print_progress(progress, total, precision, recall, output_msg):
    sys.stdout.write('\b' * len(output_msg))
    new_output_msg = 'Progress: {}/{}'.format(progress, total)
    new_output_msg += ' Precision:{:.4f}, Recall:{:.4f}'.format(precision, recall)
    sys.stdout.write(new_output_msg)
    sys.stdout.flush()

    return new_output_msg




if __name__ == '__main__':
    config = configparser.RawConfigParser()
    config.read('config.cfg')
    sys.stdout.flush()

    max_len = config.getint('data', 'max_length')
    tolerance_window = config.getint('decode', 'tolerance_window')
    tolerance_window = 4
    de_batch = config.getint('decode', 'batch_size')
    test_scp_file = config.get('data', 'test_scp')
    test_bound_file = config.get('data', 'test_bnd_file')
    corpus_type = config.get('data', 'corpus_type')
    print('=============================================================')
    print('                      Loading data                          ')
    print('=============================================================')
    sys.stdout.flush()
    data_list = load_data_into_mem(test_scp_file)
    feature_dim = len(data_list[0][0])
    total_test_utt_num = len(data_list)
    print('feature dim: ' + str(feature_dim))
    print('utt_num: ' + str(total_test_utt_num))
    config.set('data', 'feature_dim', feature_dim)

    #Load all bounds through bound file for oracle usage.
    all_bounds_list = load_all_bounds_through_file(test_bound_file, corpus_type)
    print('=============================================================')
    print('                      Set up models.                         ')
    print('=============================================================')

    sess = tf.Session()
    model = nn_model.NeuralNetwork(config, sess)
    model.restore_vars(config.get('data', 'model_loc') + '-240')

    if enable_plt == True:
        from utils.data_parser import get_specific_utt_bound
        bounds = get_specific_utt_bound(
            '/media/hdd/csie/corpus/timit/test/dr7/fdhc0/si1559.wrd')
        for bound in bounds:
            plt.axvline(x=bound, color='blue', linewidth=2)

    bound_generator = bound_loader(test_bound_file, de_batch, corpus_type)
    print('')
    output_msg = ''
    recall_list = []
    precision_list = []
    embed_num_ratio_list = []
    total_bounds = 0
    total_embeddings = 0


    counter = 0
    print('=============================================================')
    print('                      Start Decoding                         ')
    print('=============================================================')
    while counter < total_test_utt_num:
        remain_utt_num = total_test_utt_num - counter
        batch_size = min(de_batch, remain_utt_num)
        X = padding(data_list[counter:counter + batch_size], \
            max_len, feature_dim)
        seg_action = np.zeros(X.shape[:2], dtype=np.int32)
        seg_action_tmp = model.get_greedy_segmentation(X, X, seg_action, len(X))

        [rnn_code, embed_num_ratio]= model.get_tensor_val(
                                ['rnn_code', 'embed_num_ratio'],
                                 X, X, seg_action_tmp, len(X))
        embed_idx = np.sign(np.max(np.abs(rnn_code), axis=2))
        bounds_list = next(bound_generator)
        counter += len(X)
        for utt_idx in range(len(X)):
            '''
            Calculate tolerance precision and recall.
            '''
            seg_bnds = np.nonzero(embed_idx[utt_idx])[0].tolist()
            seg_bnds.sort()
            seg_bnds = seg_bnds[:-1]
            precision_list.append(\
                tolerance_precision(bounds_list[utt_idx],
                                    seg_bnds, tolerance_window))
            recall_list.append(\
                tolerance_recall(bounds_list[utt_idx],
                                 seg_bnds, tolerance_window))
            embed_num_ratio_list += [embed_num_ratio] * len(X)
            '''
            Sum up number of bounds and embeddings.
            '''

            total_bounds += len(bounds_list[utt_idx])
            total_embeddings += len(seg_bnds)

        output_msg = print_progress(counter, total_test_utt_num,
                        precision_list[counter - 1],
                        recall_list[counter - 1], output_msg)
        if enable_plt == True:
            for e_idx, e_sign in enumerate(embed_idx[1]):
                if e_sign == 1:
                    plt.axvline(x=e_idx, color='green', linewidth=2)
            [g1_gas] = model.get_tensor_val(['encoder_gate1'],
                                          X, X, oracle_seg, len(X))
            g1_gas = np.mean(g1_gas, axis=2)
            T = range(len(g1_gas[1]))
            plt.plot(T, g1_gas[1], color='red', linewidth=4)
            plt.tick_params(axis='x', labelsize=15)
            plt.tick_params(axis='y', labelsize=15)
            pdb.set_trace()
            quit()

    print('')
    title =  'precision recall f_score r_val'
    print(title)

    precision = sum(precision_list) / len(precision_list)
    recall = sum(recall_list) / len(recall_list)
    recall *= 100
    precision *= 100
    if recall == 0. or precision == 0.:
        f_score = -1.
        r_val = -1.
    else:
        f_score = (2 * precision * recall) / (precision + recall)
        r_val = r_val_eval(precision, recall)

    print('{:.4f} {:.4f} {:.4f} {:.4f}'. \
        format(precision, recall, f_score, r_val))
    print('Num. of bounds: {}; Num. of embeddings: {}'.\
        format(total_bounds, total_embeddings))
    embed_num_ratio = sum(embed_num_ratio_list) / len(embed_num_ratio_list)
    print('embed_num_ratio: {:.4f}'.format(embed_num_ratio))
