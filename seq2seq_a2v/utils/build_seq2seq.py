import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell
from .build_graph_utils import single_layer_fc
import pdb


def single_layer_rnn(cell, x, sequence_len, init_state=None):
    outputs, state = tf.nn.dynamic_rnn(cell, x,
        sequence_length=sequence_len,
        dtype=tf.float32,
        initial_state=init_state)

    return outputs, state


def gen_rnn_cell(rnn_cell_num, rnn_type, dropout_keep_prob):

    cell = tf.contrib.rnn.GRUCell(rnn_cell_num, activation=tf.nn.relu)
    #cell = tf.contrib.rnn.LayerNormBasicLSTMCell(rnn_cell_num, activation=tf.nn.relu)
    cell = tf.contrib.rnn.DropoutWrapper(cell, dropout_keep_prob)
    return cell



'''
Build seq2seq autoencoder graph.
'''
def seq2seq(hyper_parms):
    model_input = hyper_parms['model_input']
    feature_dim = hyper_parms['feature_dim']
    cell_num = hyper_parms['seq2seq_rnn_cell_num']
    rnn_type = hyper_parms['seq2seq_rnn_type']
    sequence_len = hyper_parms['sequence_len']
    encoder_rnn_layer_num = hyper_parms['encoder_rnn_layer_num']
    dropout_keep_prob = hyper_parms['dropout_keep_prob']
    batch_size = hyper_parms['batch_size']
    std_batch_size = hyper_parms['std_batch_size']
    feature_dim = hyper_parms['feature_dim']
    '''
    batch_size = tf.shape(model_input)[0]
    with tf.variable_scope('embed'):
        rnn_inputs = tf.reshape(model_input, [-1, feature_dim])
        rnn_inputs = single_layer_fc(rnn_inputs, feature_dim, 1024, activation=tf.nn.relu, scope='embed1')
        rnn_inputs = single_layer_fc(rnn_inputs, 1024, 1024, activation=tf.nn.relu, scope='embed2')
        rnn_inputs = tf.reshape(rnn_inputs, [batch_size, -1, 1024])
    '''

    rnn_inputs = model_input
    with tf.variable_scope('encoder'):
        cells = []
        for _ in range(encoder_rnn_layer_num):
            cells.append(gen_rnn_cell(cell_num, rnn_type, dropout_keep_prob))

        cell = tf.contrib.rnn.MultiRNNCell(cells)
        _, code = single_layer_rnn(cell, rnn_inputs, sequence_len)
        #rnn_code = code[-1][0]
        rnn_code = code[-1]
        code = code[-1]


    with tf.variable_scope('decoder'):
        #decoder_inputs = tf.zeros_like(rnn_inputs)
        #decoder_inputs = tf.ones_like(rnn_inputs)

        decoder_inputs = tf.tile(tf.expand_dims(code, axis=1),
            tf.stack([1, tf.cast(tf.reduce_max(sequence_len), tf.int32), 1]))

        cell = gen_rnn_cell(cell_num, rnn_type, dropout_keep_prob)
        outputs, _ = single_layer_rnn(cell, decoder_inputs,
                                      sequence_len, init_state=code)

    with tf.variable_scope('output_layer'):
        batch_size = tf.shape(outputs)[0]
        outputs = tf.reshape(outputs, [-1, cell_num])
        seq2seq_y = single_layer_fc(outputs, cell_num, feature_dim)
        seq2seq_y = tf.reshape(seq2seq_y, [batch_size, -1, feature_dim])



    return seq2seq_y, rnn_code, seq2seq_y, rnn_code


