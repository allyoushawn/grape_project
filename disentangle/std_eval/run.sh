#!/bin/bash
ans=/home/allyoushawn/Documents/data_prep/librispeech/spoken_term_detection/query/query.dev.ans
query="query_dev.pkl"
doc="doc.pkl"
cpu_num=10
rm -rf jobs
rm -rf querywise_result
for i in $( seq 0 1 315); do
    echo "python3 utils/single_query_example.py $query $doc $i $ans >>querywise_result" >>jobs
done
cat jobs | parallel --no-notice -j $cpu_num
utils/calculate_std_average_result.py querywise_result >MAP
cat MAP
