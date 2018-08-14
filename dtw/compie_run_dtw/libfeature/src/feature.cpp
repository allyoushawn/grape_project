#include <iostream>
#include <cassert>
#include "feature.h"

#define sqrt2 1.414213562373095

using namespace std;

/* Feature I/O */

bool DenseFeature::LoadFile(const string filename) {/*{{{*/
  /* Check for extension of the file */
  string ext = GetExtension(filename);

  FILE *fd = FOPEN(filename.c_str(), "r");
  if (ext == "mfc" || ext == "fbank" || ext == "plp" || ext == "gp") {
    LoadFromHtk(fd);
  } else if (ext == "feat") {
    LoadFromAscii(fd);
  } else {
    ErrorExit(__FILE__, __LINE__, 1,"Unknown extension `%s'\n", ext.c_str());
  }
  fclose(fd);
  fname_ = filename;
  return true;
}/*}}}*/

bool DenseFeature::LoadFile(const string filename, char type) {/*{{{*/
  FILE *fd = FOPEN(filename.c_str(), "r");
  bool ret = false;
  if (type == 'a') {
    ret = LoadFromAscii(fd);
  } else if (type == 'h') {
    ret = LoadFromHtk(fd);
  } else {
    assert(false);
  }
  fclose(fd);
  fname_ = filename;
  return ret;
}/*}}}*/

bool DenseFeature::WriteToAscii(FILE *fp) const {/*{{{*/

  assert(data_.nrow() > 0 && data_.ncol() > 0);

  bool ret = true;
  if (fprintf(fp, "LT=%d, LF=%d, dT=%f\n", LT(), LF(), dT_) < 0)
    ret = false;
  for (int t = 0; t < LT(); ++t) {
    if (fprintf(fp, "%.2f", data_(t, 0)) < 0) ret = false;
    for (int f = 1; f < LF(); ++f) {
      if (fprintf(fp, "\t%.2f", data_(t, f)) < 0) ret = false;
    }
    if (fprintf(fp, "\n") < 0) ret = false;
  }
  return ret;
}/*}}}*/

bool DenseFeature::WriteToHtk(FILE *fp) const {/*{{{*/

  assert(data_.nrow() > 0 && data_.ncol() > 0);

  int s_write;
  int32_t numSamp, sampPeriod;
  int16_t sampSize;

  numSamp = LT();
  sampPeriod = static_cast<int32_t>(dT_ / 1e-7);
  sampSize = 4 * LF();

  s_write = fwrite(&numSamp, 4, 1, fp);
  assert(s_write == 1);
  s_write = fwrite(&sampPeriod, 4, 1, fp);
  assert(s_write == 1);
  s_write = fwrite(&sampSize, 2, 1, fp);
  assert(s_write == 1);
  s_write = fwrite(&parmKind_, 2, 1, fp);
  assert(s_write == 1);

  for (int t = 0; t < LT(); ++t) {
    s_write = fwrite(&data_[t][0], 4, LF(), fp);
    assert(s_write == LF());
  }
  return true;
}/*}}}*/

void DenseFeature::DumpData() const {/*{{{*/
  printf("DATA:\n");
  printf("(LT, LF, dT) = (%d, %d, %g)\n", LT(), LF(), dT_);
  printf("PRIVATE:\n");
  cout << data_;
}/*}}}*/

bool DenseFeature::LoadFromHtk(FILE *fp) {/*{{{*/

  int32_t numSamp, sampPeriod;
  int16_t sampSize;
  int numDim, valSize;
  size_t s_read;

  s_read = fread(&numSamp, 4, 1, fp);
  assert(s_read == 1);
  s_read = fread(&sampPeriod, 4, 1, fp);
  assert(s_read == 1);
  s_read = fread(&sampSize, 2, 1, fp);
  assert(s_read == 1);
  s_read = fread(&parmKind_, 2, 1, fp);
  assert(s_read == 1);

  if ((02000 & parmKind_) > 0) {
    valSize = 2;
  } else {
    valSize = 4;
  }
  numDim = sampSize / valSize;

  dT_ = static_cast<float>(sampPeriod) * 1e-7;

  data_.resize(numSamp, numDim);
  /* Uncompressed type */
  if (valSize == 4) {
    if (sizeof(float) == valSize) {
      for (int t = 0; t < numSamp; ++t) {
        s_read = fread(&data_[t][0], valSize, numDim, fp);
        assert(static_cast<int>(s_read) == numDim);
      }
    } else {
      ErrorExit(__FILE__, __LINE__, 1, "sizeof(float) != 4\n");
    }
  }
  /* Compressed type */
  else if (valSize == 2) {
    int tot_val = numSamp * numDim;
    vector<int16_t> ptr(tot_val);
    s_read = fread(&ptr[0], 2, tot_val, fp);
    assert(static_cast<int>(s_read) == tot_val);

    for (int t = 0; t < numSamp; t++) {
      for (int f = 0; f < numDim; f++) {
        data_(t, f) = static_cast<float>(ptr[t * numDim + f]);
      }
    }
  }
  return true;
}/*}}}*/

bool DenseFeature::LoadFromAscii(FILE *fp) {/*{{{*/
  bool ret = true;
  int len_t, len_f;
  if (fscanf(fp, "LT=%d, LF=%d, dT=%f\n",
             &len_t, &len_f, &dT_) < 0) {
    ret = false;
  }

  data_.resize(len_t, len_f);
  for (int t = 0; t < LT(); ++t) {
    if (fscanf(fp, "%f", &data_[t][0]) != 1)
      ret = false;
    for (int f = 1; f < LF(); ++f) {
      if (fscanf(fp, "\t%f", &data_[t][f]) != 1) {
        ret = false;
      }
    }
    if (fscanf(fp, "\n") != 0) ret = false;
  }
  return ret;
}/*}}}*/


const float SparseFeature::operator()(int t, int f) const {/*{{{*/
  typeof(data_[t].begin()) itr = data_[t].find(f);
  if (itr == data_[t].end()) {
    return 0;
  } else {
    return itr->second;
  }
}/*}}}*/

