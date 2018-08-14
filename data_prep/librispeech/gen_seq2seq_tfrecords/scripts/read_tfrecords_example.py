import tensorflow as tf
import pdb

tfrecords_filename ='/home/allyoushawn/features/journal/seq2seq_ssae_ali/train.tfrecords'

#read TFRecord
def _parse_function(example_proto):
    # Define how to parse the example
    context_features = {
         'length': tf.FixedLenFeature([], dtype=tf.int64),
         'feat_dim': tf.FixedLenFeature([], dtype=tf.int64)
         }
    sequence_features = {
             'feats': tf.FixedLenSequenceFeature([], dtype=tf.float32),
         }

    # Parse the example
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
         serialized=example_proto,
         context_features=context_features,
         sequence_features=sequence_features
         )
    seq_len = context_parsed['length']
    feat_dim = context_parsed['feat_dim']
    sequence_shape1 = tf.expand_dims(seq_len, axis=0)
    sequence_shape2 = tf.expand_dims(feat_dim, axis=0)
    sequence_shape = tf.concat([sequence_shape1, sequence_shape2], axis=0)
    feats_parsed = tf.reshape(sequence_parsed['feats'], sequence_shape)
    #sequence_parsed.set_shape((context_parsed['length'], context_parsed['feat_dim']))
    return feats_parsed, tf.expand_dims(seq_len, axis=0)

def inputs(batch_size, tfrecords_filename):
    dataset = tf.data.TFRecordDataset(tfrecords_filename)
    dataset = dataset.map(_parse_function)
    #padded_shapes [None, None]: the first one means time_step_pad,
    #                            the second one means feat_dim_pad
    #dataset = dataset.padded_batch(batch_size, padded_shapes=[None, None])
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, None], [None]))
    iterator = dataset.make_initializable_iterator()
    return iterator.get_next(), iterator


batch_size = 10
batch_data, iterator = inputs(batch_size, tfrecords_filename)


sess = tf.Session()
sess.run(iterator.initializer)
tmp = sess.run(batch_data)
pdb.set_trace()
