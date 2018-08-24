#!/bin/bash

if [ -f setup.sh ]; then
  . setup.sh;
else
  echo "ERROR: setup.sh is missing!";
  exit 1;
fi

mkdir -p feat_scp


## Feature extraction
#for target in test dev query.test query.dev train; do
#    scripts/extract_feats.sh material/$target.wav.scp
#    cp $feat_loc/$target.39.cmvn.scp feat_scp/
#done

feat_loc=/home/allyoushawn/features/journal/seq2seq_ssae_ali_lambda_10_sample_10
ali_dir=material/ssae_10_sample_ali
mkdir -p $feat_loc

#for target in query.test query.dev dev test train; do
for target in test train query.dev ; do
    # 40-dim fbank feats
    #scripts/gen_tfrecords.py feat_scp/$target.cmvn.scp $feat_loc/$target.tfrecords material/$target.ctm
    # 39-dim mfcc feats
    #scripts/gen_tfrecords_ali.py feat_scp/$target.39.cmvn.scp $feat_loc/$target.tfrecords material/$target.ctm

    scripts/gen_tfrecords_with_ali.py feat_scp/$target.39.cmvn.scp $feat_loc/$target.tfrecords $ali_dir/$target.ali
    mv embed_num $feat_loc/$target.embed_num
done
