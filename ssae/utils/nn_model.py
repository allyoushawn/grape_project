import tensorflow as tf
import numpy as np
from .build_seq2seq import seq2seq
from .build_seq2seq_with_cnn import seq2seq_cnn
from .build_graph_utils import reconstruction_loss, policy_gradient_loss
from .build_policy_rnn_utils import build_policy_rnn_graph
from .build_cnn_policy import build_policy_nn, generate_vars
import pdb




class SegSeq2SeqAutoEncoder(object):
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
        self.lambda_val = config.getfloat('train', 'lambda')
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
        seq_len_filter = tf.placeholder(tf.float32, [None, None],
                                    name='seq_len_filter')


        model_input = x
        mask = tf.sign(tf.reduce_max(tf.abs(y_),
                             axis=2), name='mask')
        mask = tf.stop_gradient(mask)
        sequence_len = tf.reduce_sum(mask, axis=1,
                                     name='sequence_len')

        reversed_y_ = tf.reverse_sequence(y_,
          tf.cast(sequence_len, tf.int32), seq_axis=1,
          name='reversed_y_')

        #hyper parameters dictionary
        hyper_parms = {}
        hyper_parms['feature_dim'] = feature_dim
        hyper_parms['seq2seq_rnn_cell_num'] = seq2seq_rnn_cell_num
        hyper_parms['policy_rnn_cell_num'] = policy_rnn_cell_num
        hyper_parms['sequence_len'] = sequence_len
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
        hyper_parms['model_input'] = model_input


        with tf.variable_scope('policy_model'):
            (seg_action, policy_outputs,
            std_seg_action) = \
             build_policy_rnn_graph(hyper_parms)
             #build_policy_nn(hyper_parms)

        with tf.variable_scope('seq2seq_model'):
            seq2seq_y, self.rnn_code, \
              std_seq2seq_y, self.std_rnn_code = seq2seq(hyper_parms)

        with tf.variable_scope('embed_num_ratio'):
            embed_num = tf.sign(tf.reduce_max(
                tf.abs(self.rnn_code), axis=2))
            self.printed_embed_num_ratio = tf.reduce_sum(tf.cast(seg_action, tf.float32), axis=1)\
                / tf.reduce_sum(mask, axis=1)
            self.embed_num_ratio = tf.reduce_sum(embed_num, axis=1)\
                / tf.reduce_sum(mask, axis=1)
            self.embed_num_ratio = tf.stop_gradient(self.embed_num_ratio)

        with tf.variable_scope('loss'):
            with tf.variable_scope('reconstruction_loss'):
                self.seq2seq_re_loss = reconstruction_loss(
                                        seq2seq_y, reversed_y_, mask, utt_mask, seq_len_filter)
                std_seq2seq_re_loss = reconstruction_loss(
                                        std_seq2seq_y, reversed_y_, mask, utt_mask, seq_len_filter)
                tf.summary.scalar('seq2seq_reconstruction_loss',
                                  tf.reduce_sum(self.seq2seq_re_loss) / tf.reduce_sum(utt_mask))

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
                 tf.reduce_mean([tf.reduce_mean(tf.square(x)) for x in seq2seq_weights])

                tf.summary.scalar('seq2seq_regularization_loss',
                                  self.seq2seq_reg_loss)

            self.seq2seq_loss = tf.reduce_sum(self.seq2seq_re_loss) / tf.reduce_sum(utt_mask) + \
                                seq2seq_reg_param * self.seq2seq_reg_loss

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
            tf.summary.scalar('policy_loss', self.policy_loss)

        with tf.variable_scope('others'):
                tf.summary.scalar('embed_num_ratio',
                                  tf.reduce_sum(self.embed_num_ratio)/ tf.reduce_sum(utt_mask))
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
        self.utt_mask = utt_mask
        self.seq_len_filter = seq_len_filter

        self.tensor_list= {}
        self.tensor_list['seq2seq_re_loss'] = tf.reduce_sum(self.seq2seq_re_loss) / tf.reduce_sum(utt_mask)
        self.tensor_list['seq2seq_re_loss_ind'] = self.seq2seq_re_loss
        self.tensor_list['utt_mask'] = utt_mask
        self.tensor_list['seq2seq_y'] = seq2seq_y
        self.tensor_list['rnn_code'] = self.rnn_code
        self.tensor_list['embed_num_ratio'] = tf.reduce_sum(self.embed_num_ratio) / tf.reduce_sum(utt_mask)
        self.tensor_list['printed_embed_num_ratio'] = tf.reduce_sum(self.printed_embed_num_ratio) / tf.reduce_sum(mask)
        self.tensor_list['policy_outputs'] = policy_outputs
        self.tensor_list['reward'] = self.reward
        self.tensor_list['seg_action'] = seg_action
        self.tensor_list['std_rnn_code'] = self.std_rnn_code
        self.tensor_list['std_seg_action'] = std_seg_action
        self.tensor_list['std_seq2seq_re_loss'] = tf.reduce_sum(std_seq2seq_re_loss) / tf.reduce_sum(utt_mask)
        self.tensor_list['sequence_len'] = sequence_len
        self.tensor_list['mask'] = mask

        self.saver = tf.train.Saver()
        self.seq2seq_saver = tf.train.Saver(var_list=seq2seq_vars)

        self.policy_outputs = policy_outputs


    def init_vars(self):
        self.sess.run(tf.global_variables_initializer())


    def setup_summary(self):
        self.merged_summary = tf.summary.merge_all()


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
                encoder_capped_gvs = []
                decoder_capped_gvs = []
                for grad, var in gvs:
                    if 'policy' in var.name: continue
                    #if grad == None:
                    #    print(var.name)
                    #    seq2seq_capped_gvs.append((tf.zeros_like(var), var))
                    else:
                        seq2seq_capped_gvs.append(
                         (tf.clip_by_value(grad, -1., 1.), var))
                    if '/encoder/' in var.name:
                        encoder_capped_gvs.append((tf.clip_by_value(grad, -1., 1.), var))
                    if '/decoder/' in var.name:
                        decoder_capped_gvs.append((tf.clip_by_value(grad, -1., 1.), var))

            plot_gvs1 = [tf.reshape(x[0], [-1]) for x in encoder_capped_gvs]
            plot_gvs2 = [tf.reshape(x[0], [-1]) for x in decoder_capped_gvs]
            tf.summary.histogram('encoder_grad', tf.concat(plot_gvs1, axis=0))
            tf.summary.histogram('decoder_grad', tf.concat(plot_gvs2, axis=0))

            self.seq2seq_train_step = seq2seq_optimizer.apply_gradients(
                                                         seq2seq_capped_gvs)

            # Set up the training of policy rnn
            policy_capped_gvs = []
            policy_optimizer = tf.train.AdamOptimizer(
             learning_rate=self.policy_rnn_init_lr)
            #policy_optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.)
            with tf.variable_scope('policy_gradient_computing'):
                gvs = policy_optimizer.compute_gradients(self.policy_loss)
                for grad, var in gvs:
                    if 'seq2seq' in var.name: continue
                    #if grad == None:
                    #    print(var.name)
                    #    policy_capped_gvs.append((tf.zeros_like(var), var))
                    else:
                        policy_capped_gvs.append((tf.clip_by_value(grad, -1., 1.), var))

                # Plot gradient
                # But shold be used when reproduce_policy is set to be true.
                # However, we care about how model predict ratio% now,
                # We set reproduce_policy to be false during tensorboard's summary
                plot_gvs = [tf.reshape(x[0], [-1]) for x in policy_capped_gvs]

                tf.summary.histogram('policy_nn_grad',
                 tf.concat(plot_gvs, axis=0))

            self.policy_train_step = policy_optimizer.apply_gradients(policy_capped_gvs)



    def train_seq2seq(self, x, y_, assigned_seg_act, utt_mask, seq_len_filter):
        self.sess.run(self.seq2seq_train_step,
                 feed_dict={self.x: x, self.y_: y_,
                 self.utt_mask: utt_mask,
                 self.dropout_keep_prob: 1.0,
                 self.assigned_seg_act: assigned_seg_act,
                 self.seq_len_filter: seq_len_filter})


    def train_policy(self, x, y_, assigned_seg_act, reward, reward_baseline):
        feed = {self.x: x, self.y_: y_,
                 self.dropout_keep_prob: 1.0,
                 self.reward: reward,
                 self.assigned_seg_act: assigned_seg_act,
                 self.reward_baseline: reward_baseline,
                 self.reproduce_policy: True,
                 self.greedy_policy: False}

        self.sess.run(self.policy_train_step, feed_dict=feed)




    def update_old_prnn_vars(self):
        self.sess.run(self.old_prnn_update)


    def get_tensor_val(self, tensor_name_list, x, y_,
                       assigned_seg_act, utt_mask, seq_len_filter):
        fetch_tensors = [ self.tensor_list[tensor_name]\
                         for tensor_name in tensor_name_list ]
        return self.sess.run(fetch_tensors,
               feed_dict={self.x: x, self.y_: y_,
                 self.utt_mask: utt_mask,
                 self.dropout_keep_prob: 1.0,
                 self.assigned_seg_act: assigned_seg_act,
                 self.reproduce_policy: False,
                 self.greedy_policy: False,
                 self.seq_len_filter: seq_len_filter})


    def sample_bnds(self, x, y_, std_mode=False):
        if std_mode == False:
            fetch_tensor = self.tensor_list['seg_action']
        else:
            fetch_tensor = self.tensor_list['std_seg_action']

        return self.sess.run(fetch_tensor,
               feed_dict={self.x: x, self.y_: y_,
                 self.reproduce_policy: False,
                 self.greedy_policy: False})


    def tensorboard_summary(self, x, y_, assigned_seg_act,
                            utt_mask, reward, seq_len_filter):
        #reward = self.calculate_reward(x, y_, assigned_seg_act, batch_size)
        feed = {self.x: x, self.y_: y_,
                 self.utt_mask: utt_mask,
                 self.dropout_keep_prob: 1.0,
                 self.reward: reward,
                 self.reward_baseline: np.ones_like(reward) * np.mean(reward),
                 self.assigned_seg_act: assigned_seg_act,
                 self.reproduce_policy: False,
                 self.greedy_policy: False,
                 self.seq_len_filter: seq_len_filter}
        return self.sess.run(self.merged_summary, feed_dict=feed)


    def get_greedy_segmentation(self, x, y_, assigned_seg_act):

        return self.sess.run(self.seg_action,
                 feed_dict={self.x: x, self.y_: y_,
                 self.assigned_seg_act: assigned_seg_act,
                 self.reproduce_policy: False,
                 self.greedy_policy: True})


    def save_vars(self, path, step):
        self.saver.save(self.sess, path, global_step=step)


    def save_seq2seq_vars(self, path):
        self.seq2seq_saver.save(self.sess, path)


    def restore_vars(self, path):
        self.saver.restore(self.sess, path)


    def restore_seq2seq_vars(self, path):
        self.seq2seq_saver.restore(self.sess, path)


    def calculate_reward(self, x, y_, assigned_seg_act, utt_mask, seq_len_filter):
        re_loss, embed_num_ratio  = \
        self.sess.run([self.seq2seq_re_loss, self.embed_num_ratio],
                 feed_dict={
                 self.x: x, self.y_: y_,
                 self.utt_mask: utt_mask,
                 self.dropout_keep_prob: 1.0,
                 self.assigned_seg_act: assigned_seg_act,
                 self.seq_len_filter: seq_len_filter})

        #Calculate reward
        #re_loss_reward = (1.5 - re_loss)
        #embed_num_ratio_reward = 1. - embed_num_ratio * 5.
        re_loss_reward =  -re_loss

        #  lambda: 3 for downsampling and 5 for non-downsampling
        embed_num_ratio_reward = -embed_num_ratio * self.lambda_val
        #embed_num_ratio_reward = -embed_num_ratio * 5.
        reward = np.minimum(embed_num_ratio_reward, re_loss_reward)

        return reward



