#!/bin/bash

gpu_id=0

cur_dir=$(pwd)
dir="/media/hdd2/exps/journal_exps/downsampled_no_cmvn_filter_too_short"

python3 generate_labels.py
mv labels.tsv visualize/log

cp gen_embeds.py ${dir}
cd ${dir}
CUDA_VISIBLE_DEVICES=$gpu_id ./gen_embeds.py
cp -r plot_std ${cur_dir}

cd ${cur_dir}

cp plot_std/code.pkl visualize

cd visualize

CUDA_VISIBLE_DEVICES=$gpu_device ./visual_example.py
tensorboard --logdir log --port 6007
