import tensorflow as tf
import pdb


def generate_vars(shape):
    shape_list = [ x for x in shape]
    return tf.Variable(tf.random_uniform(shape_list, minval=-0.08, maxval=0.08))


def generate_vars_with_init(init):
    return tf.Variable(init)


def conv_layer(x, filter_shape, stride=1, padding='SAME'):
    filters = generate_vars(filter_shape)
    y = tf.nn.conv1d(x, filters, stride=stride, padding=padding)
    return tf.nn.tanh(y)


def max_pool_layer(x, pool_size=2, strides=2, padding='same'):
    return tf.layers.max_pooling1d(x, pool_size=pool_size, strides=strides,
               padding=padding, data_format='channels_first')


def policy_nn(hyper_parms, batch_size):
    x = hyper_parms['model_input']
    sequence_len = hyper_parms['sequence_len']

    conv1 = conv_layer(x, filter_shape=[21, 39, 100])
    conv1 = max_pool_layer(conv1)
    conv2 = conv_layer(conv1, filter_shape=[11, int(100 / 2), 200])
    conv2 = max_pool_layer(conv2)
    conv3 = conv_layer(conv2, filter_shape=[5, int(200 / 2), 400])
    conv3 = max_pool_layer(conv3)


    reshape1 = tf.reshape(conv3, [-1, int(400 / 2)])
    action_logits = tf.matmul(reshape1, generate_vars([int(400 / 2), 2])) + generate_vars_with_init([3.5, 0.])
    action_probs = tf.nn.softmax(action_logits)
    sampled_actions = tf.multinomial(
        tf.log(action_probs), 1, output_dtype=tf.int32)
    sampled_actions = tf.reshape(sampled_actions, [batch_size, -1, 1])
    action_probs = tf.reshape(action_probs, [batch_size, -1, 2])

    seq_mask = tf.sequence_mask(sequence_len)
    seq_mask = tf.expand_dims(seq_mask, axis=2)
    seq_mask = tf.tile(seq_mask, [1, 1, 2])
    action_probs = tf.where(seq_mask, action_probs, tf.zeros_like(action_probs))
    return action_probs, sampled_actions




def build_policy_nn(hyper_parms):

    std_batch_size = hyper_parms['std_batch_size']
    batch_size = hyper_parms['batch_size']
    sequence_len = hyper_parms['sequence_len']

    with tf.variable_scope('policy_rnn'):
        action_probs, sampled_action = policy_nn(hyper_parms, batch_size)


    with tf.variable_scope('policy_rnn', reuse=True):
        _, std_sampled_action = policy_nn(hyper_parms, std_batch_size)

    sampled_action = tf.squeeze(sampled_action, axis=2)
    std_sampled_action = tf.squeeze(std_sampled_action, axis=2)

    seq_mask = tf.sequence_mask(sequence_len)
    sampled_action = tf.where(seq_mask, sampled_action, tf.zeros_like(sampled_action))
    std_sampled_action = tf.where(seq_mask, std_sampled_action, tf.zeros_like(std_sampled_action))


    return (sampled_action, action_probs, std_sampled_action)
