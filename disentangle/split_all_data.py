#!/usr/bin/env python3
import os
import sys

if len(sys.argv) != 4:
    print('Usage: split_all_data.py <feat_dir> <seq_len> <n_spk>')
    quit()

feat_dir = sys.argv[1]
seq_len = int(sys.argv[2])
if not os.path.exists(os.path.join(feat_dir, 'all_AE')):
    os.mkdir(os.path.join(feat_dir, 'all_AE'))

feats_dir = os.path.join(feat_dir, str(seq_len))
feats_list = os.listdir(feats_dir)
feats_list = sorted(feats_list)
n_spks = len(feats_list)
print (n_spks)
n_spks_per_file = int(sys.argv[3])
n_files = n_spks // n_spks_per_file
n_files = 0

def main():
    for i in range(n_files+1):
        test_scp = os.path.join(feat_dir, 'all_AE/all_AE_'+str(i)+'.scp')
        if i != n_files:
            file_list = feats_list[n_spks_per_file*i:n_spks_per_file*(i+1)]
        if i == n_files:
            file_list = feats_list[n_spks_per_file*(i):]
        # random.shuffle(file_list)
        print ("Testing number of speakers in list "+str(i)+": " + str(len(file_list)))
        with open(test_scp, 'w') as fout:
            for f in file_list:
                fout.write(f + '\n')

if __name__=='__main__':
    main()
