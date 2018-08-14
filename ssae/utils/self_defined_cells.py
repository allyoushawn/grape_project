from tensorflow.python.ops.rnn_cell_impl import RNNCell
import tensorflow as tf
import pdb

class PolicyRNNCell(RNNCell):
    def __init__(self, rnn_cell):
        self.cell = rnn_cell


    @property
    def state_size(self):
        return self.cell.state_size


    @property
    def output_size(self):
        return 2
        #return self.cell.output_size

    def __call__(self, inputs, state, scope=None):
        output, state = self.cell(inputs, state)
        with tf.variable_scope('linear_transform'):
            w = tf.get_variable('weights', [self.cell.output_size, 2],
                   initializer=tf.random_uniform_initializer(minval=-0.08,
                                                             maxval=0.08))
            b = tf.get_variable('bias', [2],
                   initializer=tf.constant_initializer([3.5, 0.0]))

        action_logit = tf.add(tf.matmul(output, w), b)
        output = tf.nn.softmax(action_logit)

        return output, state


class DecoderRNNCell(RNNCell):
    def __init__(self, rnn_cell, batch_size):
        self.cell = rnn_cell
        self.feed_input = tf.zeros([batch_size, rnn_cell.output_size],
                              dtype=tf.float32)


    @property
    def state_size(self):
        return self.cell.state_size


    @property
    def output_size(self):
        return self.cell.output_size


    def __call__(self, inputs, state, scope=None):
        switch_embed_flag = tf.cast(tf.reduce_max(
                              tf.abs(inputs), axis=1),
                              tf.bool)
        self.feed_input = tf.where(switch_embed_flag, inputs, self.feed_input)
        output, state = self.cell(self.feed_input, state)

        return output, state


class DecoderSpeakerRNNCell(RNNCell):
    def __init__(self, rnn_cell, speaker_code, batch_size):
        self.cell = rnn_cell
        self.feed_embed = tf.zeros([batch_size, rnn_cell.output_size],
                              dtype=tf.float32)
        self.speaker_code = speaker_code


    @property
    def state_size(self):
        return self.cell.state_size


    @property
    def output_size(self):
        return self.cell.output_size


    def __call__(self, inputs, state, scope=None):
        switch_embed_flag = tf.cast(tf.reduce_max(
                              tf.abs(inputs), axis=1),
                              tf.bool)
        self.feed_embed = tf.where(switch_embed_flag, inputs, self.feed_embed)
        self.feed_input = tf.concat([self.feed_embed, self.speaker_code],
          axis=1)
        output, state = self.cell(self.feed_input, state)

        return output, state


class VAEDecoderRNNCell(RNNCell):
    def __init__(self, rnn_cell, batch_size):
        self.cell = rnn_cell
        self.feed_embed = tf.zeros([batch_size, rnn_cell.output_size / 2],
                              dtype=tf.float32)

    @property
    def state_size(self):
        return self.cell.state_size


    @property
    def output_size(self):
        return self.cell.output_size


    def __call__(self, inputs, state, scope=None):
        switch_embed_flag = tf.cast(tf.reduce_max(
                              tf.abs(inputs), axis=1),
                              tf.bool)
        self.feed_embed = tf.where(switch_embed_flag, inputs, self.feed_embed)
        output, state = self.cell(self.feed_embed, state)

        return output, state
