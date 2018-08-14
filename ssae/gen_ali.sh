#!/bin/bash
. setup.sh
dir=/home/allyoushawn/exps/ssae_lambda_10
gpu=0

## Generate .len file
#feat_file=/home/allyoushawn/features/librispeech_feats/query.dev.39.cmvn.ark
#feat-to-len ark:$feat_file ark,t:/home/allyoushawn/features/librispeech_latest_feats/query.dev.len
#exit 0

data_type=test
len_file=/home/allyoushawn/features/librispeech_latest_feats/$data_type.len
tfrecords_loc=/home/allyoushawn/features/librispeech_latest_feats
op_file=$dir/$data_type.ali

cp utils/gen_ali.py $dir
cd $dir
CUDA_VISIBLE_DEVICES=$gpu ./gen_ali.py $tfrecords_loc $data_type $len_file $op_file
