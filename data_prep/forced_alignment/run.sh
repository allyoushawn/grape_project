#!/bin/bash
. setup.sh
mkdir -p log
rm -rf log/*
bash script/prep_dict.sh
utils/prepare_lang.sh dict "<UNK>" lang_tmp lang
rm -rf lang_tmp
bash script/00.train_lm.sh || exit 1
bash script/01.format.sh || exit 1
bash script/02.extract.feat.sh | tee log/mk_mfcc.log || exit 1
bash script/03.mono.train.sh | tee log/mono_train.log || exit 1
#bash script/04a.01.mono.mkgraph.sh | tee log/mono_mkgraph.log || exit 1
#bash script/04a.02.mono.fst.sh | tee log/mono_fst.log || exit 1
bash script/05.tree.build.sh | tee log/tree_build.log || exit 1
bash script/06.tri.train.sh | tee log/tri_train.log || exit 1
bash script/perform_forced_align.sh | tee log/forced_ali.log || exit 1
#bash script/07a.01.tri.mkgraph.sh | tee log/tri_mkgraph.log || exit 1
#bash script/07a.02.tri.fst.sh | tee log/tri_decode.log || exit 1
#bash script/align_dev.sh
#CUDA_VISIBLE_DEVICES=0 bash script/08.mlp.train.sh | tee log/mlp_train.log || exit 1
#VISIBLE_DEVICES=0 bash script/09.mlp.decode.sh | tee log/mlp_decode.log || exit 1

