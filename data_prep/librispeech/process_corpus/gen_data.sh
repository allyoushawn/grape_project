#!/bin/bash
data_type=train-clean-100
librispeech=/media/hdd/csie/corpus/librispeech/$data_type

# Gen *.text
rm -f $data_type.text
for filename in $(find $librispeech -name "*.txt"); do
    cat $filename >>$data_type.text
done

# Gen *.wav.scp
rm -f $data_type.wav.scp
for filename in $(find $librispeech -name "*.flac"); do
    # Extract uttid
    tmp=`basename $filename`
    IFS='.' read -ra ADDR <<< "$tmp"
    for i in "${ADDR[@]}"; do
        echo "$i flac -c -d -s $filename |" >>$data_type.wav.scp
        break
    done
done

cat $data_type.text | sort -V >tmp.text
cat $data_type.wav.scp | sort -V >tmp.wav.scp

mv tmp.text $data_type.text
mv tmp.wav.scp $data_type.wav.scp
