from .build_graph_utils import policy_gradient_loss, log_gauss, log_normal, kld
from .build_policy_rnn_utils import build_policy_rnn_graph
from .build_seq2seq_vae import seq2seq_vae
import tensorflow as tf
import numpy as np
from .nn_model import SegSeq2SeqAutoEncoder
import pdb

class SegSeq2SeqVAE(SegSeq2SeqAutoEncoder):
    def __init__(self, config, sess):
        feature_dim = config.getint('data', 'feature_dim')
        seq2seq_rnn_cell_num = config.getint('nnet', 'seq2seq_rnn_cell_num')
        seq2seq_rnn_type = config.get('nnet', 'seq2seq_rnn_type')
        encoder_rnn_layer_num = config.getint('nnet','encoder_rnn_layer_num')
        sgd_lr = config.getfloat('train', 'sgd_lr')
        policy_rnn_cell_num = config.getint('nnet', 'policy_rnn_cell_num')
        policy_rnn_type = config.get('nnet', 'policy_rnn_type')
        policy_rnn_layer_num = config.getint('nnet', 'policy_rnn_layer_num')
        policy_rnn_reg_param = config.getfloat('train', 'prnn_l2_reg_param')
        seq2seq_reg_param = config.getfloat('train', 'seq2seq_l2_reg_param')
        self.seq2seq_init_lr = config.getfloat('train', 'seq2seq_init_lr')
        self.policy_rnn_init_lr = config.getfloat('train', 'policy_rnn_init_lr')
        batch_size = config.getint('train', 'batch_size')
        episilon = config.getfloat('train', 'episilon')
        std_batch_size = config.getint('std_eval', 'batch_size')

        #setup nn model's input tensor
        #x = tf.placeholder(tf.float32, [None, max_len, feature_dim],
        x = tf.placeholder(tf.float32, [None, None, feature_dim],
                           name='acoustic_features')
        #y_ = tf.placeholder(tf.float32, [None, max_len, feature_dim],
        y_ = tf.placeholder(tf.float32, [None, None, feature_dim],
                            name='acoustic_features')
        #batch_size = tf.placeholder(tf.int32, name='batch_size')
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        #assigned_seg_act = tf.placeholder(tf.int32, [None, max_len],
        assigned_seg_act = tf.placeholder(tf.int32, [None, None],
                                    name='assigned_seg_act')
        reward = tf.placeholder(tf.float32,[None], name='reward')
        reward_baseline = tf.placeholder(tf.float32,[None], name='reward_baseline')
        reproduce_policy = tf.placeholder(tf.bool, name='reproduce_policy')
        greedy_policy = tf.placeholder(tf.bool, name='greedy_policy')
        utt_mask = tf.placeholder(tf.float32,[None], name='utt_mask')


        model_input = x
        mask = tf.sign(tf.reduce_max(tf.abs(y_),
                             axis=2), name='mask')
        mask = tf.stop_gradient(mask)
        sequence_len = tf.reduce_sum(mask, axis=1,
                                     name='sequence_len')

        utt_mask = tf.reduce_max(mask, axis=1)
        reversed_y_ = tf.reverse_sequence(y_,
          tf.cast(sequence_len, tf.int32), seq_dim=1,
          name='reversed_y_')

        #hyper parameters dictionary
        hyper_parms = {}
        hyper_parms['feature_dim'] = feature_dim
        hyper_parms['seq2seq_rnn_cell_num'] = seq2seq_rnn_cell_num
        hyper_parms['policy_rnn_cell_num'] = policy_rnn_cell_num
        hyper_parms['sequence_len'] = sequence_len
        hyper_parms['model_input'] = model_input
        hyper_parms['mask'] = mask
        hyper_parms['dropout_keep_prob'] = dropout_keep_prob
        hyper_parms['seq2seq_rnn_type'] = seq2seq_rnn_type
        hyper_parms['policy_rnn_type'] = policy_rnn_type
        hyper_parms['reversed_y_'] = reversed_y_
        hyper_parms['encoder_rnn_layer_num'] = encoder_rnn_layer_num
        hyper_parms['policy_rnn_layer_num'] = policy_rnn_layer_num
        hyper_parms['assigned_seg_act'] = assigned_seg_act
        hyper_parms['reproduce_policy'] = reproduce_policy
        hyper_parms['greedy_policy'] = greedy_policy
        hyper_parms['batch_size'] = batch_size
        hyper_parms['std_batch_size'] = std_batch_size

        with tf.variable_scope('policy_model'):
            (seg_action, policy_outputs,
            std_seg_action) = \
             build_policy_rnn_graph(hyper_parms)

        with tf.variable_scope('seq2seq_model'):
            q_z_x_mu_z_x_log_var, x_mu_x_log_var, self.rnn_code, *_, self.std_rnn_code = \
              seq2seq_vae(hyper_parms)
            x_mu, x_log_var = tf.split(x_mu_x_log_var, 
              num_or_size_splits=2, axis=2)
            q_z_x_mu, q_z_x_log_var = tf.split(q_z_x_mu_z_x_log_var,
              num_or_size_splits=2, axis=2)

        with tf.variable_scope('embed_num_ratio'):
            embed_num = tf.sign(tf.reduce_max(
                tf.abs(self.rnn_code), axis=2))
            self.printed_embed_num_ratio = tf.reduce_sum(tf.cast(seg_action, tf.float32), axis=1)\
                / tf.reduce_sum(mask, axis=1)
            self.embed_num_ratio = tf.reduce_sum(embed_num, axis=1)\
                / tf.reduce_sum(mask, axis=1)
            self.embed_num_ratio = tf.stop_gradient(self.embed_num_ratio)

        with tf.variable_scope('loss'):
            with tf.variable_scope('vae_lower_bound'):
                log_px_z = tf.reduce_mean(tf.reduce_sum(
                  tf.tile(tf.expand_dims(mask, axis=2), [1, 1, 39]) \
                    * log_gauss(x_mu, x_log_var, y_),
                    axis=(1, 2)))
                tf.summary.scalar('log_Px_z', log_px_z)

                neg_kld = tf.reduce_mean(
                  tf.reduce_sum(-1 * kld(q_z_x_mu,q_z_x_log_var), axis=1))
                tf.summary.scalar('KL-Divergence', -1 * neg_kld)
                lower_bound = neg_kld + log_px_z

            #L2-norm regularization on policy rnn
            with tf.variable_scope('reg_loss'):
                prnn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                           scope='policy_model/policy_rnn')
                prnn_weights = [ item for item in prnn_vars if 'bias' not in item.name]
                seq2seq_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='seq2seq_model')

                #seq2seq_vars = [ item for item in seq2seq_vars if item not in prnn_vars]
                seq2seq_weights = [ item for item in seq2seq_vars if 'bias' not in item.name]

                self.seq2seq_reg_loss = \
                 tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in seq2seq_weights])

                tf.summary.scalar('seq2seq_regularization_loss',
                                  self.seq2seq_reg_loss)


            self.seq2seq_loss = -lower_bound + seq2seq_reg_param * self.seq2seq_reg_loss
            self.observed_grad = tf.gradients(self.seq2seq_loss, seq2seq_vars[0])

            tf.summary.scalar('seq2seq_loss',
                              self.seq2seq_loss)

        with tf.variable_scope('learning_rate'):
            lr_step = tf.Variable(0, trainable=False)
            self.decay_lr = tf.assign_add(lr_step, 1)
            self.lr = tf.train.exponential_decay(0.0008, lr_step, 1, 0.5)

        with tf.variable_scope('train_policy'):

            with tf.variable_scope('policy_loss'):
                self.policy_loss = \
                 policy_gradient_loss(policy_outputs, mask, assigned_seg_act,
                                      reward, reward_baseline)
                '''
                self.policy_loss = \
                 proximal_policy_gradient_loss(policy_outputs, mask,
                    assigned_seg_act, old_policy_outputs, reward, reward_baseline,
                    episilon)
                '''

            self.policy_reg_loss =  \
             tf.reduce_mean([tf.reduce_mean(tf.square(x)) for x in prnn_weights])
            self.policy_loss = self.policy_loss + \
                                policy_rnn_reg_param * self.policy_reg_loss
            tf.summary.scalar('policy_reg_loss',
                              tf.reduce_mean(self.policy_reg_loss))
            tf.summary.scalar('policy_loss', self.policy_reg_loss)

        with tf.variable_scope('others'):
                tf.summary.scalar('embed_num_ratio',
                                  tf.reduce_sum(self.printed_embed_num_ratio)/ tf.reduce_sum(utt_mask))
                #tf.summary.scalar('reward', tf.reduce_sum(reward)/ tf.reduce_sum(utt_mask))


        self.x = x
        self.y_ = y_
        self.batch_size = batch_size
        self.dropout_keep_prob = dropout_keep_prob
        self.reward = reward
        self.sess = sess
        self.config = config
        self.assigned_seg_act = assigned_seg_act
        self.sgd_lr = sgd_lr
        self.reward = reward
        self.reward_baseline = reward_baseline
        self.seg_action = seg_action
        self.reproduce_policy = reproduce_policy
        self.greedy_policy = greedy_policy
        self.lower_bound = lower_bound
        self.utt_mask = utt_mask

        self.tensor_list= {}
        self.tensor_list['utt_mask'] = utt_mask
        self.tensor_list['rnn_code'] = self.rnn_code
        self.tensor_list['embed_num_ratio'] = tf.reduce_sum(self.embed_num_ratio) / tf.reduce_sum(utt_mask)
        self.tensor_list['printed_embed_num_ratio'] = tf.reduce_sum(self.printed_embed_num_ratio) / tf.reduce_sum(mask)
        self.tensor_list['policy_outputs'] = policy_outputs
        self.tensor_list['reward'] = self.reward
        self.tensor_list['seg_action'] = seg_action
        self.tensor_list['std_rnn_code'] = self.std_rnn_code
        self.tensor_list['std_seg_action'] = std_seg_action
        self.tensor_list['sequence_len'] = sequence_len
        self.tensor_list['lower_bound'] = self.lower_bound
        self.tensor_list['mask'] = mask

        self.saver = tf.train.Saver()
        self.seq2seq_saver = tf.train.Saver(var_list=seq2seq_vars)

        self.policy_outputs = policy_outputs


    def setup_train(self):
        with tf.variable_scope('train'):

            #Set up the training of seq2seq.
            seq2seq_optimizer = tf.train.AdamOptimizer(
             learning_rate=self.seq2seq_init_lr)
            '''
            seq2seq_optimizer = tf.train.GradientDescentOptimizer(
                                  learning_rate=self.sgd_lr)
            '''
            seq2seq_capped_gvs = []
            with tf.variable_scope('gradient_computing'):
                gvs = seq2seq_optimizer.compute_gradients(self.seq2seq_loss)
                for grad, var in gvs:
                    if grad == None:
                        seq2seq_capped_gvs.append((tf.zeros_like(var), var))
                    else:
                        seq2seq_capped_gvs.append(
                         (tf.clip_by_norm(grad, 5.), var))
            self.seq2seq_train_step = seq2seq_optimizer.apply_gradients(
                                                         seq2seq_capped_gvs)

            #Set up the training of policy rnn
            policy_capped_gvs = []
            policy_optimizer = tf.train.AdamOptimizer(
             learning_rate=self.policy_rnn_init_lr)
            #policy_optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.)
            with tf.variable_scope('policy_gradient_computing'):
                gvs = policy_optimizer.compute_gradients(self.policy_loss)
                for grad, var in gvs:
                    if grad == None:
                        policy_capped_gvs.append((tf.zeros_like(var), var))
                    else:
                        policy_capped_gvs.append((tf.clip_by_norm(grad, 5.), var))
                    '''
                    #plot gradient
                    #but shold be used when reproduce_policy is set to be true.
                    #however, we care about how model predict ratio% now,
                    # we set reproduce_policy to be false during tensorboard's summary
                    tf.summary.histogram('prnn_weights_grad',
                     tf.concat([ tf.reshape(x[0], [-1]) for x in policy_capped_gvs], axis=0))
                    '''

            self.policy_train_step = policy_optimizer.apply_gradients(policy_capped_gvs)


    def calculate_reward(self, x, y_, assigned_seg_act, utt_mask):
        lower_bound, embed_num_ratio  = \
        self.sess.run([self.lower_bound, self.embed_num_ratio],
                 feed_dict={
                 self.x: x, self.y_: y_,
                 self.utt_mask: utt_mask,
                 self.dropout_keep_prob: 1.0,
                 self.assigned_seg_act: assigned_seg_act})

        #Calculate reward
        #re_loss_reward = (1.5 - re_loss)
        #embed_num_ratio_reward = 1. - embed_num_ratio * 5.
        lower_bound_reward =  lower_bound * 1.5
        embed_num_ratio_reward = -embed_num_ratio * 5.
        reward = np.minimum(embed_num_ratio_reward, lower_bound_reward)

        return reward
