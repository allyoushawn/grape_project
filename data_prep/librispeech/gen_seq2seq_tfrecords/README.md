The scripts is for generating tfrecords for ordinary audio word2vec. Each piece of data is a word. On the other hand, in the folder "gen_tfrecords", each piece of data is an utterance.
Generate tfrecords with wav.scp, ctm. If ctm is not provided, the boundary information is empty (No boundaries at all.)
