#!/bin/bash

if [ -f setup.sh ]; then
  . setup.sh;
else
  echo "ERROR: setup.sh is missing!";
  exit 1;
fi

#mkdir -p feat_scp
## Feature extraction
#for target in test dev query.test query.dev train; do
#    scripts/extract_feats.sh material/$target.wav.scp mfcc
#    #cp $feat_loc/$target.39.cmvn.scp feat_scp/
#    cp $feat_loc/$target.cmvn.scp feat_scp/
#done

for target in train dev test query.test query.dev; do
    scripts/gen_tfrecords.py feat_scp/$target.39.cmvn.scp $feat_loc/$target.tfrecords material/$target.ctm
    #scripts/gen_tfrecords.py feat_scp/$target.39.cmvn.scp $feat_loc/$target.tfrecords material/$target.ctm
    #scripts/gen_tfrecords.py feat_scp/$target.cmvn.scp $feat_loc/$target.tfrecords material/$target.ctm
done
