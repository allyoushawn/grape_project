#!/usr/bin/env python3
import sys
import pdb

if len(sys.argv) != 6:
    print('Usage: ./gen_crop_query_cmd_and_files.py <selected_query_word.txt> <query_ctm> <query_wav_scp> <output_wav_archive> <output_dir>')
    quit()

query_word_set = set()
with open(sys.argv[1]) as f:
    for line in f.readlines():
        query_word_set.add(line.rstrip())

query_uttid_wav_map = {}
with open(sys.argv[3]) as f:
    for line in f.readlines():
        tokens = line.rstrip().split()
        query_uttid_wav_map[tokens[0]] = tokens[5]

query_cmd_f = open('%s/crop_cmd.sh'%(sys.argv[5]), 'w')
query_wav_scp_f = open('%s/query.wav.scp'%(sys.argv[5]), 'w')
query_text_f = open('%s/query.text'%(sys.argv[5]), 'w')
query_idx = 0
with open(sys.argv[2]) as f:
    for line in f.readlines():
        tokens = line.rstrip().split()
        if tokens[4] not in query_word_set:  continue

        wav_loc = query_uttid_wav_map[tokens[0]]
        op_wav_loc = '%s/%s_%s.wav'%(sys.argv[4], tokens[0], str(query_idx))
        query_cmd_f.write('sox %s %s trim %s %s\n'%(wav_loc, op_wav_loc, tokens[2], tokens[3]))
        query_wav_scp_f.write('%s_%s %s\n'%(tokens[0], str(query_idx), op_wav_loc))
        query_text_f.write('%s_%s %s\n'%(tokens[0], str(query_idx), tokens[4]))
        query_idx += 1

query_cmd_f.close()
query_wav_scp_f.close()
query_text_f.close()
