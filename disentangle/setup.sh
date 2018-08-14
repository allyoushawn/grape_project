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

export seq_len=100
export feat_loc=/home/allyoushawn/features/journal/disentangle/ssae_seg
export kaldi_feat_loc=/home/allyoushawn/features/librispeech_feats
