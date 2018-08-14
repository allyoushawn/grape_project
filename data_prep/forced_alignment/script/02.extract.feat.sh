
if [ -f setup.sh ]; then
  . setup.sh;
else
  echo "ERROR: setup.sh is missing!";
  exit 1;
fi


#path=feat
path=$feat_loc
options="--use-energy=false --sample-frequency=16000"

echo "Acoustic features will be extracted in the following directory : "
echo "  $path"

mkdir -p $path
rm -rf $path/*

#tmp_dir=feat_tmp
tmp_dir=../Taiwanese_feat_tmp #For wk station
mkdir -p $tmp_dir
rm -rf $tmp_dir/*

nj=$cpu_num # 8 for 531 and 16 for wk station
for target in train dev test; do
  echo "Extracting $target set"
  log=$path/${target}.extract.log
  scp=material/${target}.wav.scp
  compute-mfcc-feats --verbose=2 $options scp:material/$target.wav.scp ark,t,scp:$path/$target.13.ark,$path/$target.13.scp 2> $log
  add-deltas scp:$path/$target.13.scp ark,t,scp:$path/$target.39.ark,$path/$target.39.scp
  compute-cmvn-stats --binary=false scp:$path/$target.39.scp ark,t:$path/$target.cmvn.results.ark
  apply-cmvn --norm-vars=false ark:$path/$target.cmvn.results.ark scp:$path/$target.39.scp ark,t,scp:$path/$target.39.cmvn.ark,$path/$target.39.cmvn.scp
done


rm -rf $tmp_dir

sec=$SECONDS

echo ""
echo "Execution time for whole script = `utils/timer.pl $sec`"
echo ""

