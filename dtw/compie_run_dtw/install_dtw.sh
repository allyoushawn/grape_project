#!/bin/bash

# Download this file (install_dtw.sh) and put it in a directory, e.g. $HOME/mydtw/
# > bash install_dtw.sh
# 
# It will download all repo from github and start compiling.
# After compilation, there are four static libraries (.a).
# libutility/lib/`uname -m`/libutility.a
# libfeature/lib/`uname -m`/libfeature.a
# libsegtree/lib/`uname -m`/libsegtree.a  
# libdtw/lib/`uname -m`/libdtw.a
# 
# The DTW example is in libdtw/test/
# > cd libdtw/test/
# > make
# There will be errors because the link to the header and library is incorrect.
# Make sure -I and -L point to the correct directory.


server="http://github.com/chunan"
repo="libutility libfeature libsegtree libdtw"

# Download repo
for r in $repo; do 
  git clone $server/$r.git
done

# Make repo
for r in $repo; do 
  make -C $r
done
