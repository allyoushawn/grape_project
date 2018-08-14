import subprocess
import random
import numpy as np
import tensorflow as tf

import pdb


def load_data_into_mem(scp_file):
    feat_proc = subprocess.Popen(['copy-feats scp:{} ark,t:- 2>/dev/null' \
                                  .format(scp_file)],stdout=subprocess.PIPE, \
                                  shell=True)
    data_list = []
    frame_list = []
    while True:
        line = feat_proc.stdout.readline().rstrip()
        if line == b'':
            feat_proc.terminate()
            break
        if b'[' in line:
            frame_list = []
            continue
        elif b']' in line:
           tmp = [float(x) for x in (line.split())[:-1]]
           frame_list.append(tmp)
           data_list.append(frame_list)
        else:
           tmp = [float(x) for x in (line.split())]
           frame_list.append(tmp)

    return data_list


def get_feat_dim(scp_file):
    feat_proc = subprocess.Popen(['copy-feats scp:{} ark,t:- 2>/dev/null' \
                                  .format(scp_file)],stdout=subprocess.PIPE, \
                                  shell=True)
    line = feat_proc.stdout.readline().rstrip()
    line = feat_proc.stdout.readline().rstrip()
    feat_dim = len(line.split())
    feat_proc.terminate()
    return feat_dim


def get_utt_num(scp_file):
    feat_proc = subprocess.Popen(['cat {} | wc -l' \
                                  .format(scp_file)],stdout=subprocess.PIPE, \
                                  shell=True)
    line = feat_proc.stdout.readline().rstrip()
    feat_proc.terminate()
    utt_num = int(line)
    return utt_num


def padding(data_list, max_len, feature_dim):
    X = np.zeros((len(data_list), max_len, feature_dim),
                 dtype = np.float32)
    for utt_idx in range(len(data_list)):
        for time_step_idx in range(min( max_len, len(data_list[utt_idx]))):
            np.copyto(X[utt_idx][time_step_idx], \
                np.array(data_list[utt_idx][time_step_idx]))
    return X

def data_loader(scp_file, batch_size, \
                total_utt_num, max_len, feature_dim):
    feat_proc = subprocess.Popen(['copy-feats scp:{} ark,t:- 2>/dev/null' \
                                  .format(scp_file)],stdout=subprocess.PIPE, \
                                  shell=True)


    #initialization
    start = False
    utt_count = 0
    utt_num = 0
    sequence_idx = 0
    #arr_size is the size of the data this time, since the size of the last batch
    #do not have to be 'batch_size'
    remain_utt_num = total_utt_num - utt_num
    arr_size = min(batch_size, remain_utt_num)
    X = np.zeros((arr_size, max_len, feature_dim),
                 dtype = np.float32)
    sequence_len = np.zeros((arr_size))

    while True:
        line = feat_proc.stdout.readline().rstrip()

        if line == '' or utt_num >= total_utt_num:
            #end of the ark, close popoen process
            feat_proc.terminate()
            yield X
            break

        if b'[' in line :
            assert(start == False)
            start = True

            processed_uttID = (line.split())[0]
            continue

        if start == True and b']' not in line:
            #features
            for idx, s in enumerate(line.split()):
                X[utt_count][sequence_idx][idx] = float(s)
            sequence_idx += 1
            continue

        if b']' in line:
            #features
            for idx, s in enumerate(line[:-1].split()):
                X[utt_count][sequence_idx][idx] = float(s)

            #The end of a utterance, reset parameters
            start = False
            sequence_idx = 0
            utt_count += 1
            utt_num += 1

            if utt_count >= batch_size:
                if utt_num >= total_utt_num:
                    feat_proc.terminate()
                yield X
                utt_count = 0
                remain_utt_num = total_utt_num - utt_num
                arr_size = min(batch_size, remain_utt_num)
                X = np.zeros((arr_size, max_len, feature_dim),
                             dtype = np.float32)
                sequence_len = np.zeros((arr_size))


def get_specific_utt_bound(path, corpus_type):
    bounds = []
    with open(path,'r') as f:
        for line in f.readlines():
            if corpus_type == 'timit':
                a_bound = int(int(line.rstrip().split()[0]) / 160)
            else:
                a_bound = int(float(line.rstrip().split()[0]) / 0.01)
            if a_bound != 0:
                bounds.append(a_bound)

    return bounds


def bound_loader(bound_scp_file, batch_size, corpus_type):
    f = open(bound_scp_file, 'r')
    bounds_list = []
    counter = 0

    while True:
        line = f.readline()
        if line == '':
            yield bounds_list
            break

        if len(bounds_list) < batch_size:
            bound_file = line.rstrip().split()[1]
            bounds = get_specific_utt_bound(bound_file, corpus_type)
            bounds_list.append(bounds)

        if len(bounds_list) >= batch_size:
            yield bounds_list
            bounds_list = []


def load_all_bounds_through_file(bound_scp_file, corpus_type):
    f = open(bound_scp_file, 'r')
    bounds_list = []
    counter = 0

    while True:
        line = f.readline()
        if line == '':
            f.close()
            return  bounds_list

        else:
            bound_file = line.rstrip().split()[1]
            bounds = get_specific_utt_bound(bound_file, corpus_type)
            bounds_list.append(bounds)


def load_all_bounds_through_file_with_mutation(bound_scp_file, data_list, mutation_prob, corpus_type):
    f = open(bound_scp_file, 'r')
    bounds_list = []
    counter = 0

    while True:
        line = f.readline()
        if line == '':
            f.close()
            return  bounds_list

        else:
            bound_file = line.rstrip().split()[1]
            bounds = get_specific_utt_bound(bound_file, corpus_type)
            '''
            sequence_len = len(data_list[counter])
            num_sampled_bounds = int(len(bounds) * (1 + mutation_prob))
            sampled_bounds = random.sample(range(sequence_len),
                                           num_sampled_bounds)
            sampled_bounds.sort()
            bounds = sampled_bounds
            '''
            bounds_list.append(bounds)
            counter += 1


def load_all_bounds_through_file_for_std(bound_scp_file, corpus_type):
    f = open(bound_scp_file, 'r')
    bounds_dict = {}
    counter = 0

    while True:
        line = f.readline()
        if line == '':
            f.close()
            return  bounds_dict

        else:
            bound_file = line.rstrip().split()[1]
            uttid = line.rstrip().split()[0]
            bounds = get_specific_utt_bound(bound_file, corpus_type)
            bounds_dict[uttid] = bounds


def read_table(table, query_scp, spoken_doc_scp, op_scp):
    label_list = []
    uttid_list = []
    query_feat_mapping = {}
    with open(query_scp, 'r') as f:
        for line in f.readlines():
            seg = line.rstrip().split()
            query_feat_mapping[seg[0]] = seg[1]

    spoken_doc_feat_mapping = {}
    with open(spoken_doc_scp, 'r') as f:
        for line in f.readlines():
            seg = line.rstrip().split()
            spoken_doc_feat_mapping[seg[0]] = seg[1]

    op_f = open(op_scp, 'w')
    with open(table, 'r') as f:
        for line in f.readlines():
            query_uttid, uttid, label = line.rstrip().split()
            label_list.append(float(label))
            op_f.write(query_uttid + ' ' + query_feat_mapping[query_uttid])
            op_f.write('\n')
            op_f.write(uttid + ' ' + spoken_doc_feat_mapping[uttid])
            op_f.write('\n')
            uttid_list.append(query_uttid)
            uttid_list.append(uttid)
    op_f.close()
    return np.array(label_list), uttid_list


def _parse_function(example_proto):
    # Define how to parse the example
    context_features = {
         'length': tf.FixedLenFeature([], dtype=tf.int64),
         'feat_dim': tf.FixedLenFeature([], dtype=tf.int64)
         }
    sequence_features = {
             'feats': tf.FixedLenSequenceFeature([], dtype=tf.float32),
         }

    # Parse the example
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
         serialized=example_proto,
         context_features=context_features,
         sequence_features=sequence_features,
         )
    seq_len = context_parsed['length']
    feat_dim = context_parsed['feat_dim']
    sequence_shape1 = tf.expand_dims(seq_len, axis=0)
    sequence_shape2 = tf.expand_dims(feat_dim, axis=0)
    sequence_shape = tf.concat([sequence_shape1, sequence_shape2], axis=0)
    feats_parsed = tf.reshape(sequence_parsed['feats'], sequence_shape)
    return feats_parsed, tf.expand_dims(seq_len, axis=0)


def read_tfrecords(feat_loc, batch_size, data_type,
                   shuffle=False, shuffle_buffer_size=3000):
    tfrecords_file = feat_loc + '/' + data_type + '.tfrecords'
    with tf.device('/cpu:0'):

        dataset = tf.data.TFRecordDataset(tfrecords_file)
        dataset = dataset.map(_parse_function, num_parallel_calls=4).prefetch(10 * batch_size)
        #padded_shapes [None, None]: the first one means time_step_pad,
        #                            the second one means feat_dim_pad
        #dataset = dataset.padded_batch(batch_size, padded_shapes=[None, None])
        if shuffle == True:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.padded_batch(batch_size, padded_shapes=([None, None], [None]))
        iterator = dataset.make_initializable_iterator()
        return iterator.get_next(), iterator
