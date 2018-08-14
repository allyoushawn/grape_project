. setup.sh

dir=exp/mono
feat_dev="cat $feat_loc/dev.39.cmvn.scp |"
compile-train-graphs $dir/tree $dir/final.mdl train/L.fst ark:train/dev.int ark:- |\
gmm-align-compiled $dir/final.mdl ark:- "scp:$feat_dev"  ark:$dir/dev.ali
