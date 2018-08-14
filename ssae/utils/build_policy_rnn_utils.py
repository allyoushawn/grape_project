import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import pdb


def single_layer_bd_rnn(cell_num, x, sequence_len, scope='rnn'):
    fw_cell = LSTMCell(cell_num / 2)
    bw_cell = LSTMCell(cell_num / 2)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        fw_cell, bw_cell, x, sequence_length=sequence_len,
        dtype=tf.float32, scope=scope)
    outputs = tf.concat(outputs, 2)
    return outputs

def single_layer_rnn(cell_num, x, sequence_len):
    cell = LSTMCell(cell_num)
    outputs, _ = tf.nn.dynamic_rnn(
        cell, x, sequence_length=sequence_len, dtype=tf.float32)
    return outputs


def generate_vars(shape):
    shape_list = [ x for x in shape]
    return tf.Variable(tf.random_uniform(shape_list,
                       minval=-0.08, maxval=0.08))


def action_prob_transform(rnn_outputs, batch_size, dim, seq_len):
    x = tf.reshape(rnn_outputs, [-1, dim])
    logits = tf.matmul(x, generate_vars([dim, 2])) + generate_vars([2])
    action_probs = tf.nn.softmax(logits)
    sampled_actions = tf.multinomial(
        tf.log(action_probs), 1, output_dtype=tf.int32)
    sampled_actions = tf.reshape(sampled_actions, [batch_size, -1, 1])
    action_probs = tf.reshape(action_probs, [batch_size, -1, 2])

    seq_mask = tf.sequence_mask(seq_len)
    seq_mask_3d = tf.expand_dims(seq_mask, axis=2)
    seq_mask_3d = tf.tile(seq_mask_3d, [1, 1, 2])
    action_probs = tf.where(seq_mask_3d,
        action_probs, tf.zeros_like(action_probs))

    sampled_actions = tf.squeeze(sampled_actions, axis=2)
    sampled_actions = tf.where(seq_mask,
        sampled_actions, tf.zeros_like(sampled_actions))

    return action_probs, sampled_actions


def policy_rnn(cell_num, x, sequence_len, batch_size):
    op = single_layer_bd_rnn(cell_num, x, sequence_len, scope='bd_1')
    op = single_layer_bd_rnn(cell_num, op, sequence_len, scope='bd_2')
    op = single_layer_rnn(cell_num, op, sequence_len)
    return op


def build_policy_rnn_graph(hyper_parms):

    mask = hyper_parms['mask']
    std_batch_size = hyper_parms['std_batch_size']
    batch_size = hyper_parms['batch_size']
    x = hyper_parms['model_input']
    cell_num = hyper_parms['policy_rnn_cell_num']
    sequence_len = hyper_parms['sequence_len']
    sequence_len = tf.cast(sequence_len, tf.int32)

    with tf.variable_scope('policy_rnn'):
        rnn_output = policy_rnn(cell_num, x, sequence_len, batch_size)
        action_prob, sampled_action = action_prob_transform(rnn_output,
            batch_size, cell_num, sequence_len)


    with tf.variable_scope('policy_rnn', reuse=True):
        rnn_output = policy_rnn(cell_num, x, sequence_len, std_batch_size)
        _, std_sampled_action = action_prob_transform(rnn_output,
            std_batch_size, cell_num, sequence_len)


    return (sampled_action, action_prob, std_sampled_action)
