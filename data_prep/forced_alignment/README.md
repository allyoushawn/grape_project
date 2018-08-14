The script is used for performing forced alignment.

To begin, you need to put required files into dir. material. The required files are {train, dev, test}.{wav.scp, text} and lexicon.txt
In setup.sh, you need to specify the kaldi path, the path where you want to put your features and the number of cpu cores to parallel lattice generation.
In path.sh, you need to specify the kaldi path the same as setup.sh

Finally, execute run.sh to train a tri-phone model to perform forced alignment.
```
./run.sh
```

