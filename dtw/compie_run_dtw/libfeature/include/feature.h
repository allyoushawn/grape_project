#ifndef _FEATURE_H_
#define _FEATURE_H_

#include <stdint.h>
#include <string>
#include <map>
#include "utility.h"

using namespace std;

class Feature {
  public:
    /* dtor */
    virtual ~Feature() {}
    /* accessors */
    virtual int LT() const = 0;
    virtual int LF() const = 0;
    virtual float dT() const { return dT_; }
    virtual string getFname() { return fname_; }
    virtual void resize(const unsigned num_samp, const unsigned num_dim) = 0;
  protected:
    Feature() {};
//    Feature(const Feature& r) {};
    virtual void Init() {
      fname_.clear();
      dT_ = 0.0;
    }
    // Data
    string fname_;
    float dT_;
    int16_t parmKind_;
};

class DenseFeature : public Feature {
  public:
    /* Ctor */
    DenseFeature() { Init(); }
    DenseFeature(const string filename) {
      Init();
      LoadFile(filename);
    }
    DenseFeature(const string filename, char type) {
      Init();
      LoadFile(filename, type);
    }
    DenseFeature(const DenseFeature &f) : Feature(f) {
      data_ = f.Data();
    }
    ~DenseFeature() {}

    /* I/O */
    bool LoadFile(const string filename);
    bool LoadFile(const string filename, char type);
    bool WriteToAscii(FILE *fp) const;
    bool WriteToHtk(FILE *fp) const;
    void DumpData() const;
    /* accessors */
    int LT() const { return data_.nrow(); }
    int LF() const { return data_.ncol(); }
    void resize(const unsigned num_samp, const unsigned num_dim) {
      data_.resize(num_samp, num_dim);
    }
    const TwoDimVector<float>& Data() const { return data_; }
    TwoDimVector<float>& Data() { return data_; }
    const float operator()(int t, int f) const { return data_(t, f); }
    float& operator()(int t, int f) { return data_(t, f); }
    const float* operator[](int t) const { return &data_[t][0]; }
    float* operator[](int t) { return &data_[t][0]; }

  private:
    bool LoadFromAscii(FILE *fp);
    bool LoadFromHtk(FILE *fp);
    // Data
    TwoDimVector<float> data_;
};

class SparseFeature : public Feature {
  public:
    /* Ctor */
    SparseFeature() { Init(); }
    SparseFeature(const string& filename) {
      Init();
      LoadFile(filename);
    }
    SparseFeature(const string& filename, char type) {
      Init();
      LoadFile(filename, type);
    }
    ~SparseFeature() {}

    /* I/O */
    bool LoadFile(const string& filename);
    bool LoadFile(const string& filename, char type);
    bool WriteToAscii(FILE *fp);
    bool WriteToHtk(FILE *fp);
    void DumpData() const;
    /* accessors */
    int LT() const { return data_.size(); }
    int LF() const { return LF_; }
    void resize(const unsigned num_samp, const unsigned num_dim) {
      data_.resize(num_samp);
      LF_ = num_dim;
    }
    const float operator()(int t, int f) const;
    const map<int, float>& operator[](int t) const { return data_[t]; }
    map<int, float>& operator[](int t) { return data_[t]; }

  private:
    bool LoadFromAscii(FILE *fp);
    bool LoadFromHtk(FILE *fp);
    // Data
    vector<map<int, float> > data_;
    int LF_;
};

#endif /*_FEATURE_H_ */
