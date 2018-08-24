#!/bin/bash


dir="/home/allyoushawn/exps/journal_exps/ssae_ali_seq2seq_dropout"
#dir="/home/allyoushawn/exps/journal_exps/oracle_seq2seq_2"

echo $dir
#exit 0

debug_mode="off"
gpu_device=0

mkdir -p $dir
rm -rf ${dir}/*

mkdir -p $dir/tensorboard/train
cp -r config.cfg train.py seg_eval.py std_eval.py $dir
cp -r utils $dir

cd $dir
mkdir models

if [ $debug_mode == "on" ]; then
    CUDA_VISIBLE_DEVICES=${gpu_device} ./train.py
    exit 0

elif [ $debug_mode == "off" ]; then
    rm -f train.log
    CUDA_VISIBLE_DEVICES=${gpu_device} ./train.py  2>&1 | tee train.log
    exit 0

    rm -f segment.result
    CUDA_VISIBLE_DEVICES=$gpu_device ./seg_eval.py 2>&1 | tee segment.result

    rm -f std.result
    CUDA_VISIBLE_DEVICES=$gpu_device ./std_eval.py 2>&1 | tee std.result
fi


