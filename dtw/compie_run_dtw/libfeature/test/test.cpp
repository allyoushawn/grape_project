#include <iostream>
#include "feature.h"

int main() {
  DenseFeature feat("data/iv1_1.mfc");
  feat.DumpData();

  feat.LoadFile("data/iv2_1.mfc");
  feat.DumpData();

  return 0;
}
