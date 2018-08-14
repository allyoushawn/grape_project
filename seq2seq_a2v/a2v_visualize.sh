#!/bin/bash

dir="/media/hdd2/exps/journal_exps/ordinary_seq2seq"
cur_dir=$(pwd)
gpu_device=1
wrd1=ATTENTION
wrd2=DIFFICULT
wrd3=CONDITION

cp restore.py ${dir}
cd $dir
CUDA_VISIBLE_DEVICES=$gpu_device ./restore.py
cp code.pkl $cur_dir/visualize
cd $cur_dir

rm -f visualize/log/labels.tsv
for i in $(seq 100); do
    echo $wrd1 >>visualize/log/labels.tsv
done
for i in $(seq 100); do
    echo $wrd2 >>visualize/log/labels.tsv
done
for i in $(seq 100); do
    echo $wrd3 >>visualize/log/labels.tsv
done

cd visualize


CUDA_VISIBLE_DEVICES=$gpu_device ./visual_example.py

tensorboard --logdir log --port 6007

