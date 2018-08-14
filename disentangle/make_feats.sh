#!/bin/bash

feat_ark=$1
prons=$2
op_feats_dir=$3

[ -f $feat_ark ] || exit 1
[ -f $prons ] || exit 1

mkdir -p $op_feats_dir

rm -f $op_feats_dir/$seq_len/*

python3 get_feat.py $prons $feat_ark $op_feats_dir

#if [ ! -f $feats_dir/extracted ]; then
#  [ -f $feats_dir ] && rm -rf $feats_dir
#  python3 get_feat.py $prons $feat_dir/cmvned_feats.ark $op_feats_dir
#  echo 1 > $feat_dir/feats/extracted
#fi
