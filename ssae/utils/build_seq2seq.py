import tensorflow as tf
from .build_graph_utils import build_dnn
from .build_graph_utils import transpose_batch_time
from tensorflow.contrib.rnn import LSTMStateTuple
from .self_defined_cells import DecoderRNNCell
from .build_cnn_policy import generate_vars
import pdb




def convert_tensor_to_ta(tensor, dtype):
    ta = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
    ta = ta.unstack(tensor)
    return ta


def gen_rnn_cell(rnn_cell_num, rnn_type, dropout_keep_prob):

    if rnn_type == 'lstm':
        cell = tf.contrib.rnn.LSTMCell(rnn_cell_num)
    else:
        cell = tf.contrib.rnn.GRUCell(rnn_cell_num)

    cell = tf.contrib.rnn.DropoutWrapper(cell, dropout_keep_prob)
    return cell


def encoder_reset_mechanism(reset, x, rnn_type):
    if rnn_type == 'lstm':
        # x: (c, h)
        emit_output = tf.where(reset, x[0], tf.zeros_like(x[0]))
        tilde_c = tf.where(reset, tf.zeros_like(x[0]), x[0])
        tilde_h = tf.where(reset, tf.zeros_like(x[1]), x[1])
        next_cell_state = LSTMStateTuple(tilde_c, tilde_h)
    else:
        emit_output = tf.where(reset, x, tf.zeros_like(x))
        next_cell_state = tf.where(reset, tf.zeros_like(x), x)

    return emit_output, next_cell_state


def decoder_reset_mechanism(reset, x, rnn_type, next_input):
    if rnn_type == 'lstm':
        # x: (c, h)
        emit_output = x[1]
        tilde_c = tf.where(reset, next_input, x[0])
        tilde_h = tf.where(reset, tf.zeros_like(x[1]), x[1])
        next_cell_state = LSTMStateTuple(tilde_c, tilde_h)
    else:
        emit_output = x
        next_cell_state = tf.where(reset, next_input, x)

    return emit_output, next_cell_state


def build_encoder_rnn(rnn_cell_num, rnn_type, rnn_inputs, assigned_seg_act,
     sequence_length, dropout_keep_prob, batch_size):

    inputs = transpose_batch_time(rnn_inputs)
    inputs_ta = convert_tensor_to_ta(inputs, tf.float32)

    seg_act = transpose_batch_time(assigned_seg_act)
    seg_act_ta = convert_tensor_to_ta(seg_act, tf.int32)

    cell = gen_rnn_cell(rnn_cell_num, rnn_type, dropout_keep_prob)

    def loop_fn(time, cell_output, cell_state, loop_state):
        # Check whether is initial condition
        next_loop_state = None
        elements_finished = (time >= tf.cast(sequence_length, tf.int32))

        # Initial condition
        if cell_output is None:  # time == 0
            next_cell_state = cell.zero_state(batch_size, tf.float32)
            emit_output = cell_output
            next_input = inputs_ta.read(time)
            return (elements_finished, next_input, next_cell_state,
                    emit_output, next_loop_state)

        else:
            # Check whether finished
            finished = tf.reduce_all(elements_finished)

            reset_state_t = tf.cond(
                finished,
                lambda: tf.zeros([batch_size], dtype=tf.int32),
                lambda: seg_act_ta.read(time - 1))

            # Last time step -> output the code
            reset_state_t = tf.where(
                tf.equal(time, tf.cast(sequence_length, tf.int32)),
                tf.ones_like(reset_state_t),
                reset_state_t)

            reset_state_t = tf.cast(reset_state_t, tf.bool)
            emit_output, next_cell_state = encoder_reset_mechanism(
                reset_state_t, cell_state, rnn_type)

            # Read next input, if it is the last time step, feed all zeros
            next_input = tf.cond(
                finished,
                lambda: tf.zeros([batch_size, rnn_inputs.get_shape()[-1]], dtype=tf.float32),
                lambda: inputs_ta.read(time))

        #send the next_input, next_cell_state to rnn cell for time_step = time operation
        return (elements_finished, next_input, next_cell_state,
                emit_output, next_loop_state)

    outputs_ta, _, _ = tf.nn.raw_rnn(cell, loop_fn)
    outputs = outputs_ta.stack()
    outputs = transpose_batch_time(outputs)
    return outputs


def build_decoder_rnn(rnn_cell_num, rnn_type, rnn_inputs,
     sequence_length, dropout_keep_prob, batch_size):

    inputs = transpose_batch_time(rnn_inputs)
    inputs_ta = convert_tensor_to_ta(inputs, tf.float32)
    cell = gen_rnn_cell(rnn_cell_num, rnn_type, dropout_keep_prob)
    cell = DecoderRNNCell(cell, batch_size)

    loop_state_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    with tf.variable_scope('linear_transform'):
        w_o = generate_vars([cell.output_size, 39])
        b_o = generate_vars([39])
    def loop_fn(time, cell_output, cell_state, loop_state):

        # Check whether finished
        elements_finished = (time >= tf.cast(sequence_length, tf.int32))
        finished = tf.reduce_all(elements_finished)

        if cell_output is None:  # time == 0
            next_cell_state = cell.zero_state(batch_size, tf.float32)
            next_input = inputs_ta.read(time)
            next_loop_state = loop_state_ta
            emit_output = cell_output
            return (elements_finished, next_input, next_cell_state,
                    emit_output, next_loop_state)
        else:

            # Read given inputs
            next_input = tf.cond(
                finished,
                lambda: tf.zeros([batch_size, rnn_inputs.get_shape()[-1]], dtype=tf.float32),
                lambda: inputs_ta.read(time))

            switch_embed_flag = tf.cast(
                tf.reduce_max(tf.abs(next_input), axis=1),
                tf.bool, name='switch_embed_flag')

            emit_output, next_cell_state = decoder_reset_mechanism(
                switch_embed_flag, cell_state, rnn_type, next_input)

            # Generate reconstruted features
            y = tf.matmul(emit_output, w_o) + b_o
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
def seq2seq(hyper_parms):
    model_input = hyper_parms['model_input']
    feature_dim = hyper_parms['feature_dim']
    seq2seq_rnn_cell_num = hyper_parms['seq2seq_rnn_cell_num']
    rnn_type = hyper_parms['seq2seq_rnn_type']
    sequence_len = hyper_parms['sequence_len']
    assigned_seg_act = hyper_parms['assigned_seg_act']
    encoder_rnn_layer_num = hyper_parms['encoder_rnn_layer_num']
    dropout_keep_prob = hyper_parms['dropout_keep_prob']
    batch_size = hyper_parms['batch_size']
    std_batch_size = hyper_parms['std_batch_size']

    with tf.variable_scope('encoder'):
        rnn_inputs = model_input
        with tf.variable_scope('embed'):
            tmp = tf.reshape(rnn_inputs, [-1, feature_dim])
            tmp = tf.matmul(tmp, generate_vars([feature_dim, 200]))
            tmp += generate_vars([200])
            tmp = tf.nn.relu(tmp)
            rnn_inputs = tf.reshape(tmp, [batch_size, -1, 200])


        rnn_code = build_encoder_rnn(seq2seq_rnn_cell_num, rnn_type, rnn_inputs, assigned_seg_act,
                     sequence_len, dropout_keep_prob, batch_size)

    with tf.variable_scope('encoder', reuse=True):
        rnn_inputs = model_input
        with tf.variable_scope('embed', reuse=True):
            tmp = tf.reshape(rnn_inputs, [-1, feature_dim])
            tmp = tf.matmul(tmp, generate_vars([feature_dim, 200]))
            tmp += generate_vars([200])
            tmp = tf.nn.relu(tmp)
            rnn_inputs = tf.reshape(tmp, [std_batch_size, -1, 200])
        std_rnn_code = build_encoder_rnn(seq2seq_rnn_cell_num, rnn_type, rnn_inputs, assigned_seg_act,
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
        seq2seq_y = build_decoder_rnn(seq2seq_rnn_cell_num, rnn_type, reversed_rnn_code,
             sequence_len, dropout_keep_prob, batch_size)
    with tf.variable_scope('decoder', reuse=True):
        std_seq2seq_y = build_decoder_rnn(seq2seq_rnn_cell_num, rnn_type, reversed_std_rnn_code,
             sequence_len, dropout_keep_prob, std_batch_size)
    return seq2seq_y, rnn_code, std_seq2seq_y, std_rnn_code


