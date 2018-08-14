#!/usr/bin/env python3
import os
from utils import *
import sys

if len(sys.argv) != 5:
    print('Usage: split_data.py <feat_dir> <seq_len> <n_files> <proportion>')
    quit()

feat_dir = sys.argv[1]
seq_len = int(sys.argv[2])
n_files = int(sys.argv[3])
proportion = float(sys.argv[4])

def main():
    split_data(feat_dir, n_files, proportion, seq_len)

if __name__=='__main__':
    main()
