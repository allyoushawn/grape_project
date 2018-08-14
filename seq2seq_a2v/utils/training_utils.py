import random
import sys
from utils.data_parser import padding
import pdb

def train_ae_grnn(model, config, data_list, all_bounds_list):

    ae_grnn_model_loc = config.get('data', 'ae_grnn_model_loc')
    max_len = config.getint('data', 'max_length')
    feature_dim = config.getint('data', 'feature_dim')
    train_utt_num = len(data_list)
    tr_batch = 60
    for epoch in range(1, 30 + 1):
        sys.stdout.write('[ AE-GRNN Training Epoch {} ]\n'.format(epoch))
        sys.stdout.flush()
        #Shuffle the data
        packed_training_data = list(zip(data_list, all_bounds_list))
        random.shuffle(packed_training_data)
        shuf_data_list, shuf_all_bounds_list = \
          zip(*packed_training_data)

        counter = 0
        output_msg = ''
        while counter < train_utt_num:
            #set up training data
            remain_utt_num = train_utt_num - counter
            batch_size = min(tr_batch, remain_utt_num)

            X = padding(shuf_data_list[counter:counter + batch_size],
                max_len, feature_dim)
            model.train_ae_grnn(X, X, 0.3, len(X))
            counter += len(X)
    model.save_ae_grnn_vars(ae_grnn_model_loc)
