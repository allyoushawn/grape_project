import os

l = 5.0
feat_dir = '/home/grtzsohalf/Desktop/LibriSpeech/query/lambda_'+str(l)
seq_len = 90
if not os.path.exists(os.path.join(feat_dir, 'all_AE')):
    os.mkdir(os.path.join(feat_dir, 'all_AE'))

feats_dir = os.path.join(feat_dir, str(seq_len))
feats_list = os.listdir(feats_dir)
feats_list = sorted(feats_list)
n_spks = len(feats_list)
print (n_spks)
# n_spks_per_file = 10
# n_files = n_spks // n_spks_per_file

def main():
    test_scp = os.path.join(feat_dir, 'all_AE/all_AE_0.scp')
    # random.shuffle(file_list)
    print ("Testing number of speakers in list "+str(0)+": " + str(len(feats_list)))
    with open(test_scp, 'w') as fout:
        for f in feats_list:
            fout.write(f + '\n')

if __name__=='__main__':
    main()
