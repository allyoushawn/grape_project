dir=exp/tri
model=$dir/final.mdl
tree=$dir/tree
lang=lang
op_ctm=op.ctm
frame_shift=0.01
print_silence=false
oov=`cat $lang/oov.int` || exit 1;


# The data that needed to be forced ali. train or dev
# If want to forced alignment test data, some path in the below script needs
# to be modified
data_type=dev


. setup.sh

# Generate ali.
feat="cat $feat_loc/$data_type.39.cmvn.scp |"
compile-train-graphs $tree $model train/L.fst ark:train/$data_type.int ark:- |\
gmm-align-compiled $model ark:- "scp:$feat"  ark:tmp.ali

# Perform forced alignment
linear-to-nbest ark:tmp.ali \
      ark:"utils/kaldi_sym2int.pl --map-oov $oov -f 2- $lang/words.txt material/$data_type.text |" '' '' ark:- | \
lattice-align-words $lang/phones/word_boundary.int $model ark:- ark:- | \
nbest-to-ctm --frame-shift=$frame_shift --print-silence=$print_silence ark:- - | \
utils/kaldi_int2sym.pl -f 5 $lang/words.txt >$op_ctm

rm tmp.ali
