kaldi=/home/allyoushawn/kaldi
export feat_loc=/media/hdd/tmp/feat
export cpu_num=4

openfst_root=$kaldi/tools/openfst
kaldi_root=$kaldi/src
PATH=$openfst_root/bin:$PATH
PATH=$kaldi_root/bin:$PATH
PATH=$kaldi_root/fstbin/:$PATH
PATH=$kaldi_root/gmmbin/:$PATH
PATH=$kaldi_root/featbin/:$PATH
PATH=$kaldi_root/sgmmbin/:$PATH
PATH=$kaldi_root/sgmm2bin/:$PATH
PATH=$kaldi_root/fgmmbin/:$PATH
PATH=$kaldi_root/latbin/:$PATH
PATH=$kaldi_root/nnetbin/:$PATH
PATH=$kaldi_root/lmbin/:$PATH
export PATH=$PATH

export dev_feat_setup="cat $feat_loc/dev.39.cmvn.scp | copy-feats scp:- ark:- |"
export test_feat_setup="cat $feat_loc/test.39.cmvn.scp | copy-feats scp:- ark:- |"
export train_feat_setup="cat $feat_loc/train.39.cmvn.scp | copy-feats scp:- ark:- |"

export dev_feat_mlp="cat $feat_loc/dev.39.cmvn.scp | "
export test_feat_mlp="cat $feat_loc/test.39.cmvn.scp | "
export train_feat_mlp="cat $feat_loc/train.39.cmvn.scp | "

