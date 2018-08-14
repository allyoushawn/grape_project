#include <iostream>
#include "utility.h"
#include "feature.h"
#include "segtree.h"

using std::cout;
using std::endl;

int main() {

  string filename = "data/N200108011200-09-14.fbank";

  /* Feature */
  DenseFeature feat(filename);
  feat.DumpData();

  /* HAC */
  SegTree segtree;
  segtree.ConstructTree(feat);
  segtree.DumpData();

  /* Acoustic segments */
  vector<int> as_index;
  float lambda = segtree.MergeMean() + 0.5 * segtree.MergeStd();
  segtree.GetBasicSeg(&as_index, lambda);
  cout << "Acoustic Segments:\n";
  for (unsigned i = 0; i < as_index.size(); ++i) {
    cout << setw(3) << as_index[i] << ": "
      << segtree.StartT(as_index[i]) << " - " << segtree.EndT(as_index[i])
      << endl;
  }

  /* Get 7 segments */
  segtree.GetBasicSeg(&as_index, 7);
  cout << "The 7 Segments:\n";
  for (unsigned i = 0; i < as_index.size(); ++i) {
    cout << setw(3) << as_index[i] << ": "
      << segtree.StartT(as_index[i]) << " - " << segtree.EndT(as_index[i])
      << endl;
  }

  return 0;
}
