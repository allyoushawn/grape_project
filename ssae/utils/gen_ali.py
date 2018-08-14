#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import configparser
import sys
import pdb
import pickle
from utils.nn_model import SegSeq2SeqAutoEncoder
from utils.data_parser import read_tfrecords




def sample_bnds(X, std_mode=False):
    seg_action = model.sample_bnds(X, X, std_mode)
    return seg_action


if __name__ == '__main__':

    config = configparser.RawConfigParser()
    config.read('config.cfg')

    #Read config parameters.
    tr_batch = config.getint('train','batch_size')
    model_loc = config.get('data', 'model_loc')
    feature_dim = config.getint('data', 'feature_dim')
    max_epoch = config.getint('train', 'max_epoch')
    std_batch = config.getint('std_eval', 'batch_size')

    feat_loc = sys.argv[1]
    data_type = sys.argv[2]
    len_file = sys.argv[3]
    op_file = sys.argv[4]

    model_num = 2500

    print('=============================================================')
    print('                      Set up models                          ')
    print('=============================================================')
    tf.reset_default_graph()
    sys.stdout.flush()

    sess = tf.Session()

    batch_data, iterator = read_tfrecords(
      feat_loc, tr_batch, data_type)

    # Set up model.
    model_type = 'seq2seq'
    if model_type != 'vae':
        model = SegSeq2SeqAutoEncoder(config, sess)
    else:
        model = SegSeq2SeqVAE(config, sess)
    # Set up training operations.
    model.setup_train()
    # Iinitialize all vars.
    model.init_vars()

    # Restore all vars.
    model.restore_vars('models/my_model-{}'.format(model_num))

    sess.run(iterator.initializer)

    # Output aligments and/or embeddings
    op_f = open(op_file, 'w')
    rnn_embeddings = []
    embedding_file_idx = 0

    counter = 0

    # Generate uttid list
    uttid_list_pool = []
    feat_len_pool = []
    with open(len_file) as f:
        for line in f.readlines():
            uttid_list_pool.append(line.rstrip().split()[0])
            feat_len_pool.append(int(line.rstrip().split()[1]))

    while True:
        try:
            X, _, seq_len = sess.run(batch_data)
            uttid_list = uttid_list_pool[counter: counter + len(X)]
            seq_len = feat_len_pool[counter: counter + len(X)]
            counter += len(X)
            if counter % (2 * tr_batch) == 0:
                print('utt: ' + str(counter))
                sys.stdout.flush()

        except tf.errors.OutOfRangeError:
            break

        # Get ali

        # Paddings
        if len(X) < tr_batch:
            paddings = np.zeros((tr_batch - len(X), X.shape[1], feature_dim),
                                dtype=float)
            input_X = np.concatenate((X, paddings), axis=0)
            utt_mask = np.ones((len(X)))
            utt_mask = np.concatenate([utt_mask, np.zeros((len(paddings)))])
        else:
            input_X = X
            utt_mask = np.ones((len(X)))

        # Sample bnds
        seg_action = sample_bnds(input_X)
        seq_len_filter = np.ones_like(seg_action)
        [rnn_code] = model.get_tensor_val(['rnn_code'],
                      input_X, input_X, seg_action, utt_mask, seq_len_filter)

        # Remove the embeddings of the padded zero arrays
        rnn_code = rnn_code[:len(X), :, :]

        # Write alignments
        segment_ali = []
        for utt_idx in range(len(rnn_code)):
            gen_embeds = rnn_code[utt_idx]
            segmented_idx_list = list(set(rnn_code[utt_idx].nonzero()[0].tolist()))
            segmented_idx_list.sort()

            # Remove the idx of the last frame
            segmented_idx_list = segmented_idx_list[:-1]

            # Add some ops here to allow upsampling
            tmp = [ idx * 4 for idx in segmented_idx_list ]
            segmented_idx_list = tmp[:]

            segment_ali.append(segmented_idx_list)
            rnn_embeddings.append(gen_embeds[~(gen_embeds==0).all(1)].tolist())
        '''
        #remove the embeddings of the padded zero arrays
        seg_action = seg_action[:len(X), :]
        segment_ali = []
        for utt_idx in range(len(seg_action)):
            segmented_idx_list = (seg_action[utt_idx].nonzero()[0]).tolist()
            segment_ali.append(segmented_idx_list)
        '''

        for uttidx, uttid in enumerate(uttid_list):
            start_frame = 0
            '''
            f.write(uttid)
            f.write(' '+ str(start_frame))
            f.write(' '+ str(segment_ali[uttidx][0]))
            f.write('\n')
            start_frame = segment_ali[uttidx][0]
            '''
            idx = 0
            while idx < len(segment_ali[uttidx]):
                if segment_ali[uttidx][idx]!= 0:
                    op_f.write(uttid)
                    op_f.write(' '+ str(start_frame))
                    op_f.write(' '+ str(segment_ali[uttidx][idx] - start_frame))
                    op_f.write('\n')
                    start_frame = segment_ali[uttidx][idx]
                idx += 1
            if len(segment_ali[uttidx]) == 0 or segment_ali[uttidx][-1] != seq_len[uttidx] - 1:
                op_f.write(uttid)
                op_f.write(' '+ str(start_frame))
                op_f.write(' '+ str(seq_len[uttidx] - start_frame))
                op_f.write('\n')

        '''
        if len(rnn_embeddings) >= 20000:
            with open('embeddings_{}.pkl'.format(embedding_file_idx), 'wb') as fp:
                pickle.dump(rnn_embeddings, fp)
            embedding_file_idx += 1
            rnn_embeddings = []
        '''

    op_f.close()
    '''
    with open('embeddings_{}.pkl'.format(embedding_file_idx), 'wb') as fp:
        pickle.dump(rnn_embeddings, fp)
    '''
