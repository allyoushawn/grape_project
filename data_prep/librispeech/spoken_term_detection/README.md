Require: {train, dev, test}.{wav.scp, text, ctm}

Generate spoken term detection dataset. Split the training set into new training set and query set. The query set is used to generate queries. The dev and test set remain untouched.

Compare query set text with the dev set and test set

Query selection based on duration more than 0.5 seconds and 5 characters.

[Dir]
    wordking_dir: contain train, dev, test set
    query: contain the information of query. *.ans contains the answers for each query, the line id is the query id and each ids in a line is the test utterance id containing the query
