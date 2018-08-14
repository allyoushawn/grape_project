embed_num_file = '/media/hdd/csie/features/journal/seq2seq/test.embed_num'
test_ctm = '/home/allyoushawn/Documents/data_prep/librispeech/spoken_term_detection/working_dir/test.ctm'
query_text = '/home/allyoushawn/Documents/data_prep/librispeech/spoken_term_detection/query/query.dev.text'


embed_num_list = []
with open(embed_num_file) as f:
    for line in f.readlines():
        embed_num_list.append(int(line.rstrip()))

uttid = 'init'
doc_idx = 0
labels = []
with open(test_ctm) as f:
    for line in f.readlines():
        line_uttid = line.rstrip().split()[0]
        if line_uttid != uttid:
            if uttid != 'init':
                embed_num = embed_num_list[doc_idx]
                try:
                    assert(embed_num == len(uttid_info) + 1)
                except:
                    import pdb
                    pdb.set_trace()
                assert(float(uttid_info[0][2]) != 0.)
                labels.append('SIL')
                for wrd_info in uttid_info[:-1]:
                    labels.append(wrd_info[4])
                labels.append('t_' + uttid_info[-1][4])
                doc_idx += 1

            uttid = line_uttid
            uttid_info = [line.rstrip().split()]

        else:
            uttid_info.append(line.rstrip().split())

# Write the last utterance info
embed_num = embed_num_list[doc_idx]
assert(embed_num == len(uttid_info) + 1)
assert(float(uttid_info[0][2]) != 0.)
labels.append('SIL')
for wrd_info in uttid_info[:-1]:
    labels.append(wrd_info[4])
labels.append('t_' + uttid_info[-1][4])


# Write query labels
counter = 0
with open(query_text) as f:
    for line in f.readlines():
        labels.append('q_' + str(counter) + '_' + line.rstrip().split()[1])
        counter += 1


op_f = open('labels.tsv', 'w')
for wrd in labels:
    op_f.write(wrd + '\n')
op_f.close()
