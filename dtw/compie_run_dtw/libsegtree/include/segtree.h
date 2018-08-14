#ifndef __SEGTREE_H__
#define __SEGTREE_H__

#include "utility.h"
#include "feature.h"

struct MergeOp {

  struct LessLoss : binary_function <MergeOp*, MergeOp*, bool> {
    bool operator() (const MergeOp* a, const MergeOp* b) const {
      return a->loss < b->loss;
    }
  };

  int lchild, rchild;
  float loss;
  float lm2; /* (length * ||mean|| ^ 2) of newly created segment */
  MergeOp* lmerge;
  MergeOp* rmerge;
};


class SegTree{
  public:
    SegTree() { Init(); }
    SegTree(char * const fname) { Load_segtree(fname); }
    void Init();
    void Load_segtree(string fname);
    void Save_segtree(string fname);
    void GetBasicSeg(vector<int>* index, float threshold);
    void GetBasicSeg(vector<int>* index, int nseg);
    void ConstructTree(const DenseFeature& feat);
    /* accessors */
    int StartT(int seg_idx) const;
    int EndT(int seg_idx) const;
    int Parent(int child_id) const;
    int Child(int parent_id, int which_child) const;
    int NumNode() const { return num_node; };
    int NumLeaf() const { return num_leaf; };
    int NumIntNode() const { return non_leaf; };
    float MergeLoss(int seg_idx) const;
    float MergeMean() const { return merge_mean; }
    float MergeStd() const { return merge_std; }
    void DumpData() const;

  private:
    void Reallocate(int n_node);
    void AssignSeg(const MergeOp& op, int t, vector<float>& lm2);
    void UpdateLossStat();
    // Data
    vector<int32_t> start_t;    // start_t[non_leaf]
    vector<int32_t> end_t;      // end_t[non_leaf]
    vector<int32_t> parent;     // parent[num_node]
    vector<int32_t> child;      // child[2 * non_leaf]
    vector<float> merge_loss;   // merge_loss[non_leaf]

    float merge_std;
    float merge_mean;

    int num_node;
    int non_leaf;
    int num_leaf; // number of frames

};
#endif
