import tensorflow as tf
from .build_graph_utils import build_dnn
from .build_graph_utils import transpose_batch_time
from tensorflow.contrib.rnn import LSTMStateTuple
from .self_defined_cells import DecoderSpeakerRNNCell
import pdb


def build_encoder_raw_rnn(rnn_cell_num, rnn_inputs, assigned_seg_act,
     sequence_length, dropout_keep_prob, batch_size):

    inputs = transpose_batch_time(rnn_inputs)
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    inputs_ta = inputs_ta.unstack(inputs)

    seg_act = transpose_batch_time(assigned_seg_act)
    seg_act_ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    seg_act_ta = seg_act_ta.unstack(seg_act)

    cell = tf.contrib.rnn.BasicLSTMCell(rnn_cell_num)
    cell = tf.contrib.rnn.DropoutWrapper(cell, dropout_keep_prob)

    def loop_fn(time, cell_output, cell_state, loop_state):
        #check whether is initial condition
        if cell_output is None:  # time == 0
            next_cell_state = cell.zero_state(batch_size, tf.float32)
        else:
            next_cell_state = cell_state
        #check whether finished
        elements_finished = (time >= tf.cast(sequence_length, tf.int32))
        finished = tf.reduce_all(elements_finished)

        if cell_output is None:
            emit_output = cell_output  # == None for time == 0

        else:

            reset_state_t = tf.cond(
                finished,
                lambda: tf.zeros([batch_size], dtype=tf.int32),
                lambda: seg_act_ta.read(time))

            #last time step -> output the code
            reset_state_t = tf.where(
                tf.equal(time, tf.cast(sequence_length, tf.int32)),
                tf.ones_like(reset_state_t),
                reset_state_t)

            emit_output = tf.where(tf.cast(reset_state_t, tf.bool),
             next_cell_state[0], tf.zeros_like(next_cell_state[0]))

            if len(cell_state) > 1: #LSTM
                tilde_cell_state = tf.where(tf.cast(reset_state_t, tf.bool),
                 tf.zeros_like(next_cell_state[0]), next_cell_state[0])
                tilde_cell_output = tf.where(tf.cast(reset_state_t, tf.bool),
                 tf.zeros_like(next_cell_state[1]), next_cell_state[1])
                next_cell_state = LSTMStateTuple(tilde_cell_state, tilde_cell_output)
            else:
                next_cell_state = tf.where(tf.cast(reset_state_t, tf.bool),
                 tf.zeros_like(next_cell_state), next_cell_state)

        next_input = tf.cond(
            finished,
            lambda: tf.zeros([batch_size, rnn_inputs.get_shape()[-1]], dtype=tf.float32),
            lambda: inputs_ta.read(time))

        next_loop_state = None
        #send the next_input, next_cell_state to rnn cell for time_step = time operation
        return (elements_finished, next_input, next_cell_state,
                emit_output, next_loop_state)

    outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
    outputs = outputs_ta.stack()
    outputs = transpose_batch_time(outputs)
    return outputs


def build_decoder_raw_rnn(rnn_cell_num, rnn_inputs, speaker_code,
      sequence_length, dropout_keep_prob, batch_size):

    inputs = transpose_batch_time(rnn_inputs)
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    inputs_ta = inputs_ta.unstack(inputs)

    cell = tf.contrib.rnn.BasicLSTMCell(rnn_cell_num)
    cell = tf.contrib.rnn.DropoutWrapper(cell, dropout_keep_prob)
    cell = DecoderSpeakerRNNCell(cell, speaker_code, batch_size)

    #the tensor array will ask for a size = max_len's zero array
    loop_state_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    def loop_fn(time, cell_output, cell_state, loop_state):
        #check whether is initial condition
        if cell_output is None:  # time == 0
            next_cell_state = cell.zero_state(batch_size, tf.float32)
        else:
            next_cell_state = cell_state
        #check whether finished
        elements_finished = (time >= tf.cast(sequence_length, tf.int32))
        finished = tf.reduce_all(elements_finished)

        #read given inputs
        next_input = tf.cond(
            finished,
            lambda: tf.zeros([batch_size, rnn_inputs.get_shape()[-1]], dtype=tf.float32),
            lambda: inputs_ta.read(time))

        switch_embed_flag = tf.cast(tf.reduce_max(
                              tf.abs(next_input), axis=1),
                              tf.bool, name='switch_embed_flag')

        if cell_output is not None: # time > 0
            if len(cell_state) > 1: #LSTM case
                tilde_cell_state = tf.where(switch_embed_flag,
                 tf.zeros_like(next_input), next_cell_state[0])
                tilde_cell_output = tf.where(tf.cast(switch_embed_flag, tf.bool),
                 tf.zeros_like(next_cell_state[1]), next_cell_state[1])
                next_cell_state = LSTMStateTuple(tilde_cell_state, tilde_cell_output)
            else: #GRU case
                next_cell_state = tf.where(switch_embed_flag,
                 tf.zeros_like(next_input), next_cell_state)

        #generate reconstruted features
        if cell_output == None:
            next_cell_state = LSTMStateTuple(next_input, next_cell_state[1])
            with tf.variable_scope('linear_transform'):
                w_o = tf.get_variable('weights', [cell.output_size, 39],\
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
                b_o = tf.get_variable('bias', [39],\
                  initializer=tf.constant_initializer(0.1))
        else:
            with tf.variable_scope('linear_transform', reuse=True):
                w_o = tf.get_variable('weights', [cell.output_size, 39],\
                  initializer=tf.truncated_normal_initializer(stddev=0.1))
                b_o = tf.get_variable('bias', [39],\
                  initializer=tf.constant_initializer(0.1))

        emit_output = cell_output
        if cell_output == None: # time == 0
            next_loop_state = loop_state_ta
        else:
            y = tf.add(tf.matmul(cell_output, w_o), b_o, name='reconstruction')
            next_loop_state = loop_state.write(time - 1, y)

        #next_input = tf.zeros_like(next_input)
        return (elements_finished, next_input, next_cell_state,
                emit_output, next_loop_state)

    *_, loop_state_ta = tf.nn.raw_rnn(cell, loop_fn)
    y = loop_state_ta.stack()
    y = transpose_batch_time(y)

    return y


'''
Build seq2seq autoencoder graph.
'''
def seq2seq_cnn(hyper_parms):
    model_input = hyper_parms['model_input']
    feature_dim = hyper_parms['feature_dim']
    seq2seq_rnn_cell_num = hyper_parms['seq2seq_rnn_cell_num']
    seq2seq_rnn_type = hyper_parms['seq2seq_rnn_type']
    policy_rnn_cell_num = hyper_parms['policy_rnn_cell_num']
    policy_rnn_type = hyper_parms['policy_rnn_type']
    sequence_len = hyper_parms['sequence_len']
    assigned_seg_act = hyper_parms['assigned_seg_act']
    encoder_rnn_layer_num = hyper_parms['encoder_rnn_layer_num']
    policy_rnn_layer_num = hyper_parms['policy_rnn_layer_num']
    dropout_keep_prob = hyper_parms['dropout_keep_prob']
    batch_size = hyper_parms['batch_size']
    std_batch_size = hyper_parms['std_batch_size']

    width = 5
    stride = 3
    max_pool_size = 3
    speaker_code_dim = 100
    filters = tf.Variable(
     tf.truncated_normal([width, feature_dim, speaker_code_dim], stddev=0.1))
    conv_model_output = tf.nn.conv1d(model_input, filters, stride=stride, padding='VALID')
    conv_model_output = tf.nn.relu(conv_model_output)
    max_pool_model_input = tf.layers.max_pooling1d(conv_model_output, pool_size=max_pool_size, strides=1)
    conv_len = tf.floor((sequence_len - tf.floor(width / 2)) / stride) - tf.floor(max_pool_size / 2) - 1
    conv_mask = tf.sequence_mask(conv_len, dtype=max_pool_model_input.dtype)
    conv_mask = tf.tile(tf.expand_dims(conv_mask, axis=2), [1, 1, speaker_code_dim])
    speaker_code = tf.reduce_sum(max_pool_model_input * conv_mask, axis=1) / \
      tf.reduce_sum(conv_mask, axis=1)

    with tf.variable_scope('encoder'):
        rnn_inputs = model_input
        rnn_code = build_encoder_raw_rnn(seq2seq_rnn_cell_num, rnn_inputs, assigned_seg_act,
                     sequence_len, dropout_keep_prob, batch_size)

    with tf.variable_scope('encoder', reuse=True):
        rnn_inputs = model_input
        std_rnn_code = build_encoder_raw_rnn(seq2seq_rnn_cell_num, rnn_inputs, assigned_seg_act,
                     sequence_len, dropout_keep_prob, std_batch_size)

    #reverse the code and answer
    reversed_rnn_code = tf.reverse_sequence(rnn_code,
      tf.cast(sequence_len, tf.int32), seq_dim=1,
      name='reversed_rnn_code')
    #reverse the code and answer
    reversed_std_rnn_code = tf.reverse_sequence(std_rnn_code,
      tf.cast(sequence_len, tf.int32), seq_dim=1,
      name='reversed_std_rnn_code')

    with tf.variable_scope('decoder'):
        seq2seq_y = build_decoder_raw_rnn(
          seq2seq_rnn_cell_num, reversed_rnn_code, speaker_code,
          sequence_len, dropout_keep_prob, batch_size)
    with tf.variable_scope('decoder', reuse=True):
        std_seq2seq_y = build_decoder_raw_rnn(
          seq2seq_rnn_cell_num, reversed_std_rnn_code, speaker_code,
          sequence_len, dropout_keep_prob, std_batch_size)
    return seq2seq_y, rnn_code, std_seq2seq_y, std_rnn_code


