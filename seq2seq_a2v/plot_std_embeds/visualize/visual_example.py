#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import pickle

with open('code.pkl', 'rb') as fp:
    audio_vec = pickle.load(fp)
embedding_dim = len(audio_vec[0])

embedding_placeholder = tf.placeholder(tf.float32, [len(audio_vec), embedding_dim])


# Create randomly initialized embedding weights which will be trained.
N = len(audio_vec) # Number of items (vocab size).
D = embedding_dim # Dimensionality of the embedding.
embedding_var = tf.Variable(tf.random_normal([N,D]), name='word_embedding')
embedding_init = embedding_var.assign(embedding_placeholder)

# Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
# Link this tensor to its metadata file (e.g. labels).
#embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
LOG_DIR = 'log'
embedding.metadata_path = 'labels.tsv'

# Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.summary.FileWriter(LOG_DIR)

# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# read this file during startup.
projector.visualize_embeddings(summary_writer, config)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(embedding_init, feed_dict={embedding_placeholder: audio_vec})
saver.save(sess, 'log/my_model')
