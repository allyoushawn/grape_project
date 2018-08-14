#!/usr/bin/env python3
import subprocess
import tensorflow as tf
import pdb
import sys





# Write TFRecord
def make_example(sequence):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    # Downsampling
    down_sampled_sequence = []
    for idx, element in enumerate(sequence):
        if idx % 4 == 0:
            down_sampled_sequence.append(element)

    sequence = down_sampled_sequence

    try:
        feat_dim = len(sequence[0])
    except:
        pdb.set_trace()
    sequence_length = len(sequence)



    ex.context.feature['length'].int64_list.value.append(sequence_length)
    ex.context.feature['feat_dim'].int64_list.value.append(feat_dim)

    # Feature lists for the two sequential features of our example
    fl_tokens = ex.feature_lists.feature_list['feats']

    for feat in sequence:
        for token in feat:
            fl_tokens.feature.add().float_list.value.append(token)
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


def split_utterance_into_words_write_into_tfrecords(feats_sequence, bnd_sequence, len_limit, writer, gen_embed_num_file_f):

    counter = 0
    if len(bnd_sequence) == 0:
        if len(feats_sequence) == 0: pdb.set_trace()
        ex = make_example(feats_sequence)
        writer.write(ex.SerializeToString())
        counter += 1
    else:
        start_idx = 0
        for bnd in bnd_sequence:
            if bnd == 0: continue
            if bnd - start_idx > len_limit:
                ex = make_example(feats_sequence[start_idx: start_idx + len_limit])
                writer.write(ex.SerializeToString())
                counter += 1
                start_idx = bnd
                continue

            if len(feats_sequence[start_idx:bnd]) == 0: pdb.set_trace()
            ex = make_example(feats_sequence[start_idx: bnd])
            writer.write(ex.SerializeToString())
            counter += 1
            start_idx = bnd
        if start_idx < len(feats_sequence) - 1:
            if bnd - start_idx <= len_limit:
                if len(feats_sequence[start_idx:]) == 0: pdb.set_trace()
                ex = make_example(feats_sequence[start_idx:])
                writer.write(ex.SerializeToString())
                counter += 1

    gen_embed_num_file_f.write(str(counter) + '\n')




if __name__ == '__main__':

    if len(sys.argv) != 4 and len(sys.argv) != 3:
        print('Usage: ./gen_tfrecords.py <scp_file> <output_tfrecords_filename> [ctm_file]')

    scp_file = sys.argv[1]
    tfrecords_filename = sys.argv[2]
    if len(sys.argv) == 4:
        ctm_file = sys.argv[3]
    else:
        ctm_file = None
    gen_embed_num_file_f = open('embed_num', 'w')

    '''
    scp_file = 'feat_scp/test.39.cmvn.scp'
    ctm_file = 'material/test.ctm'
    tfrecords_filename = 'tmp.tfrecords'
    '''

    len_limit = 100

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
    op_f = open('validate.bnd', 'w')

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
            '''
            ex = make_example(feats_sequence, bnd_sequence, len_limit)
            writer.write(ex.SerializeToString())
            '''
            if len(feats_sequence) > 40:
                split_utterance_into_words_write_into_tfrecords(feats_sequence, bnd_sequence, len_limit, writer, gen_embed_num_file_f)
            feats_sequence = []
            uttid = next_uttid
            bnd_sequence = next_bnd_sequence

        else:
            # Features: x1, x2, x3, ..., xd
            feats_sequence.append([ float(x) for x in line.rstrip().split()])

    op_f.close()

    if ctm_fp != None:
        ctm_fp.close()
    writer.close()
    gen_embed_num_file_f.close()

