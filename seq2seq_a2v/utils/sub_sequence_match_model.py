#!/usr/bin/env python3
import tensorflow as tf

class SubSequenceMatchModel(object):
    def __init__(self, sess):
        self.doc_embed = tf.placeholder(tf.float32)
        self.query_embed = tf.placeholder(tf.float32)
        normed_doc_embed = tf.nn.l2_normalize(self.doc_embed, 2)
        normed_query_embed = tf.nn.l2_normalize(self.query_embed, 1)
        #if number of embed of query is more than doc, the score should be zero
        #or, use the conv2d to compute the score
        sub_match_result = tf.nn.conv2d(normed_doc_embed, normed_query_embed, [1,1,1,1], 'VALID')
        self.sub_match_result = tf.squeeze(sub_match_result, [0,2,3])
        self.sess = sess

    def match_score(self, doc_embed, query_embed):
        return self.sess.run(self.sub_match_result, feed_dict={self.doc_embed: doc_embed, self.query_embed: query_embed})
