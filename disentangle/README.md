execute run.sh

To perform STD evaluation, first generate .pkl file for query and doc. Each pickle file is a list of numpy array. Each element in the list is a[ Ni * feat_dim ] shape numpy array where Ni is the number of embeddings for the single query (or doc). The length of the list should be the number of query or the number of doc.

After generating the .pkl file, put them into the directory std_dir and switch to the directory. Then perform ./run.sh would perform STD evaluation. To parallel the evaluation process, you can specify the cpu_num in run.sh
