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
        seq2seq_reg_param = config.getfloat('train', 'seq2seq_l2_reg_param')
        self.seq2seq_init_lr = config.getfloat('train', 'seq2seq_init_lr')
        batch_size = config.getint('train', 'batch_size')
        std_batch_size = config.getint('std_eval', 'batch_size')


        #setup nn model's input tensor
        x = tf.placeholder(tf.float32, [None, None, feature_dim],
                           name='acoustic_features')
        y_ = tf.placeholder(tf.float32, [None, None, feature_dim],
                            name='acoustic_features')
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        utt_mask = tf.placeholder(tf.float32,[None], name='utt_mask')


        model_input = x
        mask = tf.sign(tf.reduce_max(tf.abs(y_),
                             axis=2), name='mask')
        mask = tf.stop_gradient(mask)
        sequence_len = tf.reduce_sum(mask, axis=1,
                                     name='sequence_len')


        #hyper parameters dictionary
        hyper_parms = {}
        hyper_parms['feature_dim'] = feature_dim
        hyper_parms['seq2seq_rnn_cell_num'] = seq2seq_rnn_cell_num
        hyper_parms['sequence_len'] = sequence_len
        hyper_parms['mask'] = mask
        hyper_parms['dropout_keep_prob'] = dropout_keep_prob
        hyper_parms['seq2seq_rnn_type'] = seq2seq_rnn_type
        hyper_parms['encoder_rnn_layer_num'] = encoder_rnn_layer_num
        hyper_parms['batch_size'] = batch_size
        hyper_parms['std_batch_size'] = std_batch_size
        hyper_parms['model_input'] = model_input



        with tf.variable_scope('seq2seq_model'):
            seq2seq_y, self.rnn_code, \
              std_seq2seq_y, self.std_rnn_code = seq2seq(hyper_parms)


        with tf.variable_scope('loss'):
            with tf.variable_scope('reconstruction_loss'):
                self.seq2seq_re_loss = reconstruction_loss(
                                        seq2seq_y, y_, mask, utt_mask)
                std_seq2seq_re_loss = reconstruction_loss(
                                        std_seq2seq_y, y_, mask, utt_mask)
                tf.summary.scalar('seq2seq_reconstruction_loss',
                                  tf.reduce_sum(self.seq2seq_re_loss) / tf.reduce_sum(utt_mask))

            #L2-norm regularization on policy rnn
            with tf.variable_scope('reg_loss'):
                seq2seq_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='seq2seq_model')

                #seq2seq_vars = [ item for item in seq2seq_vars if item not in prnn_vars]
                seq2seq_weights = [ item for item in seq2seq_vars if 'bias' not in item.name]

                self.seq2seq_reg_loss = \
                 tf.reduce_mean([tf.reduce_mean(tf.square(x)) for x in seq2seq_weights])

                tf.summary.scalar('seq2seq_regularization_loss',
                                  self.seq2seq_reg_loss)

            with tf.variable_scope('sim_loss'):
                normed_code = tf.nn.l2_normalize(self.rnn_code, axis=1)
                part1_rnn_code = normed_code[1:]
                part1_rnn_code = tf.concat([part1_rnn_code, normed_code[:1]], axis=0)
                sim_loss = tf.abs(1 - tf.losses.cosine_distance(part1_rnn_code, normed_code, axis=1))
                tf.summary.scalar('cosine_similarity', sim_loss)

            with tf.variable_scope('norm_loss'):
                l2_norm = tf.reduce_mean(tf.norm(self.rnn_code, axis=1))
                tf.summary.scalar('l2 norm', l2_norm)

            with tf.variable_scope('partial_sparsity_loss'):
                psl_code = (self.rnn_code + 1.) / 2
                psl_loss = tf.reduce_mean(psl_code * (1. - psl_code))
                tf.summary.scalar('psl_loss', psl_loss)

            #self.seq2seq_re_loss *= (sequence_len / 25.)
            self.seq2seq_loss = tf.reduce_sum(self.seq2seq_re_loss) / tf.reduce_sum(utt_mask) + \
                                seq2seq_reg_param * self.seq2seq_reg_loss + \
                                0.01 * sim_loss + 0.01 * tf.abs(l2_norm - 1.) \
                                + 0.0 * psl_loss

            self.observed_grad = tf.gradients(self.seq2seq_loss, seq2seq_vars[0])

            tf.summary.scalar('seq2seq_loss',
                              self.seq2seq_loss)

        with tf.variable_scope('learning_rate'):
            lr_step = tf.Variable(0, trainable=False)
            self.decay_lr = tf.assign_add(lr_step, 1)
            self.lr = tf.train.exponential_decay(0.0008, lr_step, 1, 0.5)



        self.x = x
        self.y_ = y_
        self.batch_size = batch_size
        self.dropout_keep_prob = dropout_keep_prob
        self.sess = sess
        self.config = config
        self.sgd_lr = sgd_lr
        self.utt_mask = utt_mask

        self.tensor_list= {}
        self.tensor_list['seq2seq_re_loss'] = tf.reduce_sum(self.seq2seq_re_loss) / tf.reduce_sum(utt_mask)
        self.tensor_list['seq2seq_re_loss_ind'] = self.seq2seq_re_loss
        self.tensor_list['utt_mask'] = utt_mask
        self.tensor_list['seq2seq_y'] = seq2seq_y
        self.tensor_list['rnn_code'] = self.rnn_code
        self.tensor_list['std_rnn_code'] = self.std_rnn_code
        self.tensor_list['std_seq2seq_re_loss'] = tf.reduce_sum(std_seq2seq_re_loss) / tf.reduce_sum(utt_mask)
        self.tensor_list['sequence_len'] = sequence_len
        self.tensor_list['mask'] = mask

        self.saver = tf.train.Saver()
        self.seq2seq_saver = tf.train.Saver(var_list=seq2seq_vars)



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




    def train_seq2seq(self, x, y_, utt_mask):
        self.sess.run(self.seq2seq_train_step,
                 feed_dict={self.x: x, self.y_: y_,
                 self.utt_mask: utt_mask,
                 self.dropout_keep_prob: 0.5})



    def get_tensor_val(self, tensor_name_list, x, y_, utt_mask):
        fetch_tensors = [ self.tensor_list[tensor_name]\
                         for tensor_name in tensor_name_list ]
        return self.sess.run(fetch_tensors,
               feed_dict={self.x: x, self.y_: y_,
                 self.utt_mask: utt_mask,
                 self.dropout_keep_prob: 1.0})


    def tensorboard_summary(self, x, y_, utt_mask):
        feed = {self.x: x, self.y_: y_,
                 self.utt_mask: utt_mask,
                 self.dropout_keep_prob: 1.0}
        return self.sess.run(self.merged_summary, feed_dict=feed)



    def save_vars(self, path, step):
        self.saver.save(self.sess, path, global_step=step)


    def save_seq2seq_vars(self, path):
        self.seq2seq_saver.save(self.sess, path)


    def restore_vars(self, path):
        self.saver.restore(self.sess, path)


    def restore_seq2seq_vars(self, path):
        self.seq2seq_saver.restore(self.sess, path)





