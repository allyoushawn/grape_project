#!/usr/bin/env python3
import numpy as np
import struct

def write_feature(row_feature, file, period=100000):
    nN, nF = np.shape(row_feature)
    fout = open(file, 'wb')
    fout.write(struct.pack('<i', nN))
    fout.write(struct.pack('<i', period))
    fout.write(struct.pack('<h', nF*4))
    fout.write(struct.pack('<h', 9))
    for i in range(nN):
        for j in range(nF):
            fout.write(struct.pack('<f', row_feature[i, j]))
    fout.close()



ark_f = open('/media/hdd/csie/features/librispeech_feats/query.dev.39.cmvn.ark', 'r')

separate_dir = '/media/hdd/csie/features/librispeech_feats/separate_mfcc_dir'
data_list = []
while True:
    line = ark_f.readline().rstrip()
    if line == '':
        break
    if '[' in line:
        uttid = line.split()[0]
        data_list = []
        continue
    elif ']' in line:
        data_list.append( [ float(x) for x in line.split()[:-1]])
        write_feature(np.array(data_list), separate_dir + '/'+ uttid + '.mfc')
    else:
        data_list.append( [ float(x) for x in line.split()])
ark_f.close()

ark_f = open('/media/hdd/csie/features/librispeech_feats/test.39.cmvn.ark', 'r')
data_list = []
while True:
    line = ark_f.readline().rstrip()
    if line == '':
        break
    if '[' in line:
        uttid = line.split()[0]
        data_list = []
        continue
    elif ']' in line:
        data_list.append( [ float(x) for x in line.split()[:-1]])
        write_feature(np.array(data_list), separate_dir + '/'+ uttid + '.mfc')
    else:
        data_list.append( [ float(x) for x in line.split()])
ark_f.close()
