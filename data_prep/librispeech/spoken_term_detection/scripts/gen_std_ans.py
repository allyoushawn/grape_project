#!/usr/bin/env python3
import sys

if len(sys.argv) != 4:
    print('Usage: ./gen_std_ans.py <query_text> <doc_text> <output_ans>')
    quit()

op_f = open(sys.argv[3], 'w')

doc_contents = []
with open(sys.argv[2]) as doc_f:
    for line in doc_f.readlines():
        doc_contents.append(line.rstrip())

with open(sys.argv[1]) as f:
    for line in f.readlines():
        query_word = line.rstrip().split()[1]
        doc_idx = 0
        for doc in doc_contents:
            if query_word in doc:
                op_f.write(str(doc_idx) + ' ')
            doc_idx += 1
        op_f.write('\n')

op_f.close()
