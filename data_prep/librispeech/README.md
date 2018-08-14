The order should be as follows:

process_corpus: generate {train, dev, test}.{wav.scp, .text} and lexicon.txt
forced_alignment: generate .ctm files
spoken_term_detection: split traininig set into query set and new training set; query selection
gen_tfrecords: feature extraction, convert them into tfrecords file
