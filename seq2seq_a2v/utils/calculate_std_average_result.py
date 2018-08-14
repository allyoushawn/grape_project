#!/usr/bin/env python3
import sys

if len(sys.argv) != 2:
    print('Usage: calculate_std_average_result.py <result>')
    quit()

result = sys.argv[1]
scores = []
with open(result) as fp:
    for line in fp.readlines():
        scores.append(float(line.rstrip().split()[0]))

print('{:.4f}'.format(sum(scores) / len(scores)))
