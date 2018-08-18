#!/bin/bash

kaldi_root=/opt/kaldi
kaldi_src=$kaldi_root/src


openfst_root=$kaldi_root/tools/openfst
PATH=$openfst_root/bin:$PATH
PATH=$kaldi_src/bin:$PATH
PATH=$kaldi_src/fstbin/:$PATH
PATH=$kaldi_src/gmmbin/:$PATH
PATH=$kaldi_src/featbin/:$PATH
PATH=$kaldi_src/sgmmbin/:$PATH
PATH=$kaldi_src/sgmm2bin/:$PATH
PATH=$kaldi_src/fgmmbin/:$PATH
PATH=$kaldi_src/latbin/:$PATH
PATH=$kaldi_src/nnetbin/:$PATH
export PATH=$PATH

# Use the following command to show the longest length of the alignments
#cat alignments/ssae_ali/test.ali | sort -V -k 3 | tail
export seq_len=600
export feat_loc=/home/allyoushawn/features/journal/disentangle/ssae_ali_seg
export kaldi_feat_loc=/home/allyoushawn/features/librispeech_feats
