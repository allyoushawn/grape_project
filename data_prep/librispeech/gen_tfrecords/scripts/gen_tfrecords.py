#!/usr/bin/env python3
import subprocess
import tensorflow as tf
import pdb
import sys





# Write TFRecord
def make_example(sequence, bnd_list, len_limit, scp_file):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example

    # Downsampling
    down_sampled_sequence = []
    down_sampled_bnd_list = []
    for idx, element in enumerate(sequence):
        if idx % 4 == 0:
            down_sampled_sequence.append(element)
    for bnd in bnd_list:
        bnd = int(bnd / 4)
        down_sampled_bnd_list.append(bnd)

    sequence = down_sampled_sequence
    bnd_list = down_sampled_bnd_list

    feat_dim = len(sequence[0])
    # Only have length limit on training data
    if 'train' in scp_file:
        sequence_length = min(len(sequence), len_limit)

        if sequence_length < len(sequence):
            sequence = sequence[:sequence_length]
    else:
        sequence_length = len(sequence)

    bnds_arr = [0] * sequence_length
    for bnd in bnd_list:
        if bnd >= sequence_length: break
        bnds_arr[bnd] = 1

    ex.context.feature['length'].int64_list.value.append(sequence_length)
    ex.context.feature['feat_dim'].int64_list.value.append(feat_dim)

    # Feature lists for the two sequential features of our example
    fl_tokens = ex.feature_lists.feature_list['feats']
    bnds_tokens = ex.feature_lists.feature_list['bnds']

    for feat in sequence:
        for token in feat:
            fl_tokens.feature.add().float_list.value.append(token)
    for bnd in bnds_arr:
        bnds_tokens.feature.add().int64_list.value.append(bnd)
    return ex


def gen_bnd_seq_with_ctm(uttid, bnd_sequence, ctm_fp=None):

    if ctm_fp == None:
        return [], None, []


    while True:
        line = ctm_fp.readline()
        if line == '':    break

        if line.rstrip().split()[0] != uttid:
            new_bnd_sequence = [int(float(line.rstrip().split()[2]) * 100)]
            new_uttid = line.rstrip().split()[0]
            return bnd_sequence, new_uttid, new_bnd_sequence

        else:
            bnd_sequence.append(int(float(line.rstrip().split()[2]) * 100))

    return bnd_sequence, None, None


if __name__ == '__main__':

    if len(sys.argv) != 4 and len(sys.argv) != 3:
        print('Usage: ./gen_tfrecords.py <scp_file> <output_tfrecords_filename> [ctm_file]')

    scp_file = sys.argv[1]
    tfrecords_filename = sys.argv[2]
    if len(sys.argv) == 4:
        ctm_file = sys.argv[3]
    else:
        ctm_file = None

    '''
    scp_file = 'feat_scp/test.39.cmvn.scp'
    ctm_file = 'material/test.ctm'
    tfrecords_filename = 'tmp.tfrecords'
    '''

    len_limit = 1000

    feat_proc = subprocess.Popen(['copy-feats scp:{} ark,t:- 2>/dev/null' \
                                  .format(scp_file)],stdout=subprocess.PIPE, \
                                  shell=True)
    try:
        ctm_fp = open(ctm_file)
    except:
        print('ctm does not exists, generate tfrecords without boundary information')
        ctm_fp = None

    uttid = 'init'
    feats_sequence = []
    bnd_sequence = []

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    while True:
        line = feat_proc.stdout.readline().rstrip()

        if line == b'':
            # End of the ark, close popoen process
            feat_proc.terminate()
            break

        if b'[' in line :
            if ctm_fp == None:
                next_uttid = uttid
                next_bnd_sequence = bnd_sequence
                continue

            # uttid [
            if uttid == 'init':
                uttid = line.split()[0].decode('utf-8')
            else:
                try:
                    assert(uttid == (line.split())[0].decode('utf-8'))
                except:
                    pdb.set_trace()

            #print(uttid)
            #if uttid == '8975-270782-0114':
            #    pdb.set_trace()
            bnd_sequence, next_uttid, next_bnd_sequence = \
                gen_bnd_seq_with_ctm(uttid, bnd_sequence, ctm_fp)

            '''
            uttid = (line.split())[0]
            # Covert from bytes to str
            uttid = uttid.decode('utf-8')
            continue
            '''

        elif b']' in line:
            # Features: x1, x2, x3, ... xd ]
            feats_sequence.append([ float(x) for x in line[:-1].rstrip().split()])
            if bnd_sequence == None:
                pdb.set_trace()

            ex = make_example(feats_sequence, bnd_sequence, len_limit, scp_file)
            writer.write(ex.SerializeToString())
            feats_sequence = []
            uttid = next_uttid
            bnd_sequence = next_bnd_sequence

        else:
            # Features: x1, x2, x3, ..., xd
            feats_sequence.append([ float(x) for x in line.rstrip().split()])

    if ctm_fp != None:
        ctm_fp.close()
    writer.close()
