
if [ -f setup.sh ]; then
  . setup.sh;
else
  echo "ERROR: setup.sh is missing!";
  exit 1;
fi

if [ $#  -ne 2 ]; then
    echo "Usage: extract_feats.sh <wav_scp> <feat type>"
fi

scp=$1
feat_type=$2
path=$feat_loc
options="--use-energy=false --sample-frequency=16000"


if [ "$feat_type" = "fbank" ]; then
     options="$options --num-mel-bins=40"
fi

echo "Acoustic features will be extracted in the following directory : "
echo "  $path"

mkdir -p $path

tmp=`basename $scp`
echo "  Extracting with $scp"
target=${tmp%.wav.scp}

log=$path/${target}.extract.log


if [ "$feat_type" = "mfcc" ]; then
    compute-mfcc-feats --verbose=2 $options scp:$scp ark,t,scp:$path/$target.13.ark,$path/$target.13.scp 2> $log
    add-deltas scp:$path/$target.13.scp ark,t,scp:$path/$target.39.ark,$path/$target.39.scp
    compute-cmvn-stats --binary=false scp:$path/$target.39.scp ark,t:$path/$target.cmvn.results.ark
    apply-cmvn --norm-vars=true ark:$path/$target.cmvn.results.ark scp:$path/$target.39.scp ark,t,scp:$path/$target.39.cmvn.ark,$path/$target.39.cmvn.scp
fi


if [ "$feat_type" = "fbank" ]; then
    compute-fbank-feats --verbose=2 $options scp:$scp ark,t,scp:$path/$target.ark,$path/$target.scp 2> $log
    compute-cmvn-stats --binary=false scp:$path/$target.scp ark,t:$path/$target.cmvn.results.ark
    apply-cmvn --norm-vars=true ark:$path/$target.cmvn.results.ark scp:$path/$target.scp ark,t,scp:$path/$target.cmvn.ark,$path/$target.cmvn.scp
fi
