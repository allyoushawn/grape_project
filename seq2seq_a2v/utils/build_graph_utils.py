import tensorflow as tf
from  tensorflow.python.ops import nn_ops
import numpy as np
import pdb


'''
Build a layer of dnn, that is , a fully-connected (fc) layer.
'''
def single_layer_fc(x, input_dim, output_dim, activation=None, scope='fc'):
    with tf.variable_scope(scope):
        w = tf.get_variable('weights', [input_dim, output_dim],
               initializer=tf.random_uniform_initializer(minval=-0.08,
                                                         maxval=0.08))
        b = tf.get_variable('bias', [output_dim],
               initializer=tf.random_uniform_initializer(minval=-0.08,
                                                         maxval=0.08))

        # activation = None -> Linear
        if activation == None:
            return tf.matmul(x, w) + b
        else:
            return activation(tf.matmul(x, w) + b)

'''
Build a layer of dnn, that is , a fully-connected (fc) layer.
'''
def build_dnn(x, input_dim, output_dim):
    w = tf.Variable(tf.truncated_normal([input_dim, output_dim],
                                        stddev=0.1), name='weights')
    b = tf.Variable(tf.constant(0.1, shape=[output_dim]), name='bias')
    return tf.nn.relu(tf.matmul(x, w) + b, name='activation')


'''
Zoneout implementation.
'''
def zoneout(x_candidate, x, keep_prob):
    new_x = keep_prob * nn_ops.dropout(x_candidate - x, keep_prob) + x

    return new_x


'''
Compute the masked reconstruction loss.
'''
def reconstruction_loss(y, y_, mask, utt_mask):
    #square_error = tf.multiply(tf.subtract(y_, y), tf.subtract(y_, y))
    square_error = tf.losses.mean_squared_error(y_, y, reduction=tf.losses.Reduction.NONE)
    square_error = tf.reduce_mean(square_error, axis=2)
    #square_error = tf.clip_by_value(tf.sqrt(square_error), 1e-10, 1.0)
    square_error = tf.sqrt(square_error)
    square_error *= mask

    sum_square_error = tf.reduce_sum(square_error, axis=1)
    sum_square_error_base = tf.reduce_sum(mask, axis=1)
    sum_square_error_base = tf.where(tf.cast(utt_mask, tf.bool),
      sum_square_error_base, tf.ones_like(sum_square_error_base))
    sum_square_error /= sum_square_error_base

    return sum_square_error * utt_mask


'''
Compute the loss for policy gradient.
'''
def policy_gradient_loss(y, mask, exe_action, reward, reward_baseline):
    utt_mask = tf.reduce_max(mask, axis=1)
    prediction = tf.one_hot(tf.cast(exe_action, tf.int32), 2)
    xent_loss = -1 * tf.reduce_sum(prediction * \
                      tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=2)
    xent_loss *= mask
    xent_loss = tf.reduce_sum(xent_loss, axis=1)
    xent_loss /= tf.reduce_sum(mask, axis=1)
    policy_loss = xent_loss * (reward - reward_baseline)
    policy_loss = tf.reduce_sum(policy_loss) / tf.reduce_sum(utt_mask)
    return policy_loss


'''
Compute the loss for proximal policy gradient.
'''
def proximal_policy_gradient_loss(pi, mask, action_id_sequence, pi_old,
                                  reward, reward_baseline, episilon):
    action_sequence = tf.one_hot(tf.cast(action_id_sequence, tf.int32), 2)
    prob_ratio = pi / pi_old
    prob_ratio *= action_sequence
    prob_ratio = tf.reduce_sum(prob_ratio, axis=2)
    prob_ratio *= mask
    prob_ratio = tf.reduce_sum(prob_ratio, axis=1)
    prob_ratio /= tf.reduce_sum(mask, axis=1)
    #sur_obj = prob_ratio * (reward - reward_baseline)
    '''
    clip_prob_ratio = tf.where(tf.greater((reward - reward_baseline), 0.),
                               tf.maximum(prob_ratio, 1 + episilon),
                               tf.maximum(prob_ratio, 1 - episilon))
    '''
    clip_prob_ratio = tf.clip_by_value(prob_ratio, 1 - episilon, 1 + episilon)
    sur_obj = tf.minimum(prob_ratio * (reward - reward_baseline),
                     clip_prob_ratio * (reward - reward_baseline))
    sur_obj *= -1
    return sur_obj


def transpose_batch_time(x):
  """Transpose the batch and time dimensions of a Tensor.

  Retains as much of the static shape information as possible.

  Args:
    x: A tensor of rank 2 or higher.

  Returns:
    x transposed along the first two dimensions.

  Raises:
    ValueError: if `x` is rank 1 or lower.
  """
  x_static_shape = x.get_shape()
  if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
    raise ValueError(
        "Expected input tensor %s to have rank at least 2, but saw shape: %s" %
        (x, x_static_shape))
  x_rank = tf.rank(x)
  x_t = tf.transpose(
      x, tf.concat(
          ([1, 0], tf.range(2, x_rank)), axis=0))
  x_t.set_shape(
      tf.TensorShape([
          x_static_shape[1].value, x_static_shape[0].value
      ]).concatenate(x_static_shape[2:]))
  return x_t


def kld(mu, logvar, q_mu=None, q_logvar=None):
    """compute dimension-wise KL-divergence
    -0.5 (1 + logvar - q_logvar - (exp(logvar) + (mu - q_mu)^2) / exp(q_logvar))
    q_mu, q_logvar assumed 0 is set to None
    """
    if q_mu is None:
        q_mu = tf.zeros_like(mu)
    else:
        print("using non-default q_mu %s" % q_mu)

    if q_logvar is None:
        q_logvar = tf.zeros_like(logvar)
    else:
        print("using non-default q_logvar %s" % q_logvar)

    if isinstance(mu, tf.Tensor):
        mu_shape = mu.get_shape().as_list()
    else:
        mu_shape = list(np.asarray(mu).shape)

    if isinstance(q_mu, tf.Tensor):
        q_mu_shape = q_mu.get_shape().as_list()
    else:
        q_mu_shape = list(np.asarray(q_mu).shape)

    if isinstance(logvar, tf.Tensor):
        logvar_shape = logvar.get_shape().as_list()
    else:
        logvar_shape = list(np.asarray(logvar).shape)

    if isinstance(q_logvar, tf.Tensor):
        q_logvar_shape = q_logvar.get_shape().as_list()
    else:
        q_logvar_shape = list(np.asarray(q_logvar).shape)

    if not np.all(mu_shape == logvar_shape):
        raise ValueError("mu_shape (%s) and logvar_shape (%s) does not match" % (
            mu_shape, logvar_shape))
    if not np.all(mu_shape == q_mu_shape):
        raise ValueError("mu_shape (%s) and q_mu_shape (%s) does not match" % (
            mu_shape, q_mu_shape))
    if not np.all(mu_shape == q_logvar_shape):
        raise ValueError("mu_shape (%s) and q_logvar_shape (%s) does not match" % (
            mu_shape, q_logvar_shape))

    return -0.5 * (1 + logvar - q_logvar - \
            (tf.pow(mu - q_mu, 2) + tf.exp(logvar)) / tf.exp(q_logvar))


def log_gauss(mu, logvar, x):
    """compute point-wise log prob of Gaussian"""
    x_shape = x.get_shape().as_list()

    if isinstance(mu, tf.Tensor):
        mu_shape = mu.get_shape().as_list()
    else:
        mu_shape = list(np.asarray(mu).shape)

    if isinstance(logvar, tf.Tensor):
        logvar_shape = logvar.get_shape().as_list()
    else:
        logvar_shape = list(np.asarray(logvar).shape)

    '''
    if not np.all(x_shape == mu_shape):
        raise ValueError("x_shape (%s) and mu_shape (%s) does not match" % (
            x_shape, mu_shape))
    if not np.all(x_shape == logvar_shape):
        raise ValueError("x_shape (%s) and logvar_shape (%s) does not match" % (
            x_shape, logvar_shape))
    '''

    return -0.5 * (np.log(2 * np.pi) + logvar + tf.pow((x - mu), 2) / tf.exp(logvar))

def log_normal(x):
    """compute point-wise log prob of Gaussian"""
    return -0.5 * (np.log(2 * np.pi) + tf.pow(x, 2))
