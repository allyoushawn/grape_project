#!/usr/bin/env python3
import sys

if len(sys.argv) != 3:
    print('Usage: no_alignment_filter.py <ctm_file> <file>')

uttid_set = set()
with open(sys.argv[1]) as f:
    for line in f.readlines():
        uttid_set.add(line.rstrip().split()[0])

with open(sys.argv[2]) as f:
    for line in f.readlines():
        uttid = line.rstrip().split()[0]
        if uttid not in uttid_set:
            continue
        else:
            print(line.rstrip())
