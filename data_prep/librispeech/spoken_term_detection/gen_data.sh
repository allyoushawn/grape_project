#!/bin/bash
. setup.sh
mkdir -p working_dir

# Filter out utterances without alignments
scripts/no_alignment_filter.py material/dev_clean.ctm material/dev-clean.text >working_dir/dev.text
scripts/no_alignment_filter.py material/dev_clean.ctm material/dev-clean.wav.scp >working_dir/dev.wav.scp
cp material/dev_clean.ctm working_dir/dev.ctm
scripts/no_alignment_filter.py material/test_clean.ctm material/test-clean.text >working_dir/test.text
scripts/no_alignment_filter.py material/test_clean.ctm material/test-clean.wav.scp >working_dir/test.wav.scp
cp material/test_clean.ctm working_dir/test.ctm
scripts/no_alignment_filter.py material/train_clean_100.ctm material/train-clean-100.text >working_dir/train.text.full
scripts/no_alignment_filter.py material/train_clean_100.ctm material/train-clean-100.wav.scp >working_dir/train.wav.scp.full
cp material/train_clean_100.ctm working_dir/train.ctm.full



cat working_dir/train.text.full | sed -n 1~3p >working_dir/query.text
cat working_dir/train.wav.scp.full | sed -n 1~3p >working_dir/query.wav.scp
cat working_dir/train.text.full | sed -e 1~3d >working_dir/train.text
cat working_dir/train.wav.scp.full | sed -e 1~3d >working_dir/train.wav.scp


#Gen train.ctm and query.ctm
scripts/split_train_full_ctm.py

# Select query words
scripts/gen_query_word.py >qualified_query_word.txt

cat qualified_query_word.txt | shuf | head -n 40 >selected_query_word.txt
rm -f qualified_query_word.txt


mkdir -p query
mkdir -p $query_wav_archive
rm -f $query_wav_archive/*
# Generate query wav
# This part may be different for different corpus
scripts/gen_crop_query_cmd_and_files.py selected_query_word.txt working_dir/query.ctm \
    working_dir/query.wav.scp $query_wav_archive query

bash query/crop_cmd.sh
rm -f query/crop_cmd.sh

mv selected_query_word.txt query

cat query/query.text | sed -e 1~2d >query/query.test.text
cat query/query.text | sed -n 1~2p >query/query.dev.text
cat query/query.wav.scp | sed -e 1~2d >query/query.test.wav.scp
cat query/query.wav.scp | sed -n 1~2p >query/query.dev.wav.scp
rm query/query.text query/query.wav.scp


scripts/gen_std_ans.py query/query.test.text working_dir/test.text query/query.test.ans
scripts/gen_std_ans.py query/query.dev.text working_dir/test.text query/query.dev.ans

rm -f working_dir/*.full working_dir/query.*
