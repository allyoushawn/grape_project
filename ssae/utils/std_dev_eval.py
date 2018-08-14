#!/usr/bin/env python3
import subprocess
import sys
import pickle

if len(sys.argv) != 3:
    print('Usage: std_dev_eval.py <std_dir> <ans>')
    quit()

std_dir=sys.argv[1]
std_ans=sys.argv[2]

subprocess.call('rm -f {}/result'.format(std_dir), shell=True)

op_f = open('{}/jobs'.format(std_dir), 'w')

command = 'utils/single_query_example.py'
query = std_dir + '/query.pkl'
doc = std_dir + '/doc.pkl'

with open(query, 'rb') as fp:
    query_num = len(pickle.load(fp))

for idx in range(query_num):
    op_f.write('{} {} {} {} {} >>{}/querywise_result\n'.format(
                command, query, doc, idx, std_ans, std_dir))
op_f.close()

subprocess.call('cat {}/jobs | parallel --no-notice -j 4 '.format(std_dir), shell=True)
subprocess.call('rm {}/*.pkl'.format(std_dir), shell=True)
subprocess.call('utils/calculate_std_average_result.py {}/querywise_result >{}/MAP'.format(std_dir, std_dir), shell=True)
