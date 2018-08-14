#include <iostream>
#include <cassert>
#include <vector>
#include <list>
#include <deque>
#include <cmath>
#include <numeric>
#include <functional>
#include <algorithm>
#include "segtree.h"

const int sizeFloat = sizeof(float);

void SegTree::Init() {/*{{{*/
  merge_mean = 0.0;
  merge_std = 0.0;
  num_node  = 0;
  non_leaf  = 0;
  num_leaf  = 0;
}/*}}}*/

void SegTree::Reallocate(int n_node) {/*{{{*/

  num_node = n_node;
  num_leaf = (num_node + 1) / 2;
  non_leaf = num_node - num_leaf;

  start_t.resize(non_leaf);
  end_t.resize(non_leaf);
  parent.resize(num_node);
  child.resize(2 * non_leaf);
  merge_loss.resize(non_leaf);

}/*}}}*/

void SegTree::Load_segtree(string fname) {/*{{{*/

  FILE *fd = FOPEN(fname.c_str(), "r");
  int ret;
  int32_t n_node;
  /* num_node, non_leaf, num_leaf, child */
  ret = fread(&n_node, 4, 1, fd);
  assert(ret == 1);

  Reallocate(n_node);

  ret = fread(&start_t[0], 4, non_leaf, fd);
  assert(ret == non_leaf);

  ret = fread(&end_t[0], 4, non_leaf, fd);
  assert(ret == non_leaf);

  ret = fread(&parent[0], 4, num_node, fd);
  assert(ret == num_node);

  ret = fread(&child[0], 4, 2 * non_leaf, fd);
  assert(ret == 2 * non_leaf);

  ret = fread(&merge_loss[0], sizeFloat, non_leaf, fd);
  assert(ret == non_leaf);

  fclose(fd);

  //cout << "file std = " << merge_std << endl;

  UpdateLossStat();

}/*}}}*/

void SegTree::UpdateLossStat() {/*{{{*/

  /* Calculate Mean & Std */
  double sum = 0.0;
  double sqr_sum = 0.0;
  for (int i = 0; i < non_leaf; ++i) {
    sum += merge_loss[i];
    sqr_sum += merge_loss[i] * merge_loss[i];
  }
  merge_mean = sum / non_leaf;
  merge_std = sqrt(sqr_sum / non_leaf - merge_mean * merge_mean);

  /* Eliminate too large mergings */
  int n_merge = non_leaf;
  double cutoff = merge_mean + 10.0 * merge_std;
  for (int i = non_leaf - 1; i >= 0; --i) {
    if (merge_loss[i] > cutoff) {
      n_merge--;
      sum -= merge_loss[i];
      sqr_sum -= merge_loss[i] * merge_loss[i];
    }
  }
  merge_mean = sum / n_merge;
  merge_std = sqrt(sqr_sum / n_merge - merge_mean * merge_mean);

}/*}}}*/


void SegTree::Save_segtree(string fname) {/*{{{*/

  FILE *fd = FOPEN(fname.c_str(), "w");
  int ret;
  int32_t n_node;
  /* num_node, non_leaf, num_leaf, child */
  ret = fread(&n_node, 4, 1, fd);
  assert(ret == 1);

  ret = fwrite(&start_t[0], 4, non_leaf, fd);
  assert(ret == non_leaf);

  ret = fwrite(&end_t[0], 4, non_leaf, fd);
  assert(ret == non_leaf);

  ret = fwrite(&parent[0], 4, num_node, fd);
  assert(ret == num_node);

  ret = fwrite(&child[0], 4, 2 * non_leaf, fd);
  assert(ret == 2 * non_leaf);

  ret = fwrite(&merge_loss[0], sizeFloat, non_leaf, fd);
  assert(ret == non_leaf);

  fclose(fd);

}/*}}}*/



float AddThenSquare(float a, float b) { return (a + b) * (a + b); }

float SubThenSquare(float a, float b) { return (a - b) * (a - b); }

float CalLm2(TwoDimVector<float>& accum_feat, int s, int e) {/*{{{*/
  float len = e - s + 1;
  vector<float> v = accum_feat[e];

  if (s > 0) {
    for (unsigned i =0; i < v.size(); ++i) {
      v[i] -= accum_feat(s - 1, i);
    }
  }
  return inner_product(v.begin(), v.end(), v.begin(), 0.0f) / len;
}/*}}}*/

void SegTree::AssignSeg(const MergeOp& op, int t, vector<float>& lm2) {/*{{{*/

  int seg_idx = num_leaf + t;

  start_t[t] = StartT(op.lchild);
  end_t[t] = EndT(op.rchild);
  parent[op.lchild] = seg_idx;
  parent[op.rchild] = seg_idx;
  child[2 * t] = op.lchild;
  child[2 * t + 1] = op.rchild;
  lm2[seg_idx] = op.lm2;
  merge_loss[t] = max<float>(MergeLoss(seg_idx - 1), op.loss);

}/*}}}*/

void SegTree::ConstructTree(const DenseFeature& feat) {/*{{{*/

  Reallocate(2 * feat.LT() - 1);

  /* Calculate (-length * ||mean|| ^ 2) for each frame */
  vector<float> lm2(num_node);
  for (int t = 0; t < num_leaf; ++t) {
    lm2[t] = -inner_product(feat.Data()[t].begin(), feat.Data()[t].end(),
                            feat.Data()[t].begin(), 0.0f);
  }

  /* Calculate initial segments and push into heap */
  list<MergeOp*> heap;
  MergeOp* prev_op = NULL;
  for (int t = 0; t < non_leaf; ++t) {
    MergeOp* op = new MergeOp;
    op->lchild = t;
    op->rchild = t + 1;
    op->lm2 = -0.5 * inner_product(
        feat.Data()[t].begin(), feat.Data()[t].end(),
        feat.Data()[t + 1].begin(), 0.0f, plus<float>(), AddThenSquare);
    op->loss = op->lm2 - lm2[t] - lm2[t + 1];
    op->lmerge = prev_op;
    op->rmerge = NULL;
    heap.push_back(op);
    if (prev_op != NULL) {
      prev_op->rmerge = op;
    }
    prev_op= op;
  }

  /* Calculate accumulated feature so mean vector can be calculated in O(1) */
  TwoDimVector<float> accum_feat = feat.Data();
  for (int t = 1; t < feat.LT(); ++t) {
    for (int i = 0; i < feat.LF(); ++i) {
      accum_feat(t, i) += accum_feat(t - 1, i);
    }
  }

  /* HAC */
  for (int t = 0; t < non_leaf; ++t) {
    list<MergeOp*>::iterator target;
    target = min_element(heap.begin(), heap.end(), MergeOp::LessLoss());
    MergeOp* mop = *target;
    AssignSeg(*mop, t, lm2);
    heap.erase(target);

    /* Change left neighbor merge */
    if (mop->lmerge != NULL) {
      MergeOp& lop = *(mop->lmerge);
      lop.rchild = num_leaf + t;
      lop.rmerge = mop->rmerge;
      lop.lm2 = -CalLm2(accum_feat, StartT(lop.lchild), EndT(lop.rchild));
      lop.loss = lop.lm2 - lm2[lop.lchild] - lm2[lop.rchild];
    }

    /* Change right neighbor merge */
    if (mop->rmerge != NULL) {
      MergeOp& rop = *(mop->rmerge);
      rop.lchild = num_leaf + t;
      rop.lmerge = mop->lmerge;
      rop.lm2 = -CalLm2(accum_feat, StartT(rop.lchild), EndT(rop.rchild));
      rop.loss = rop.lm2 - lm2[rop.lchild] - lm2[rop.rchild];
    }

    /* Delete merged op at this iteration */
    delete mop;
  }
  UpdateLossStat();

}/*}}}*/



void SegTree::DumpData() const {/*{{{*/

  printf("Parent: \n");
  for (int i = 0; i < num_node; ++i) {
    printf("Parent[%3d] = %3d\n", i, Parent(i));
  }
  printf("Children: \n");
  for (int i = NumLeaf(); i < NumNode(); ++i) {
    printf("Child[%3d] = {%3d, %3d}\n", i, Child(i, 0), Child(i, 1));
  }
  printf("Mergeloss: \n");
  for (int i = NumLeaf(); i < NumNode(); ++i) {
    printf("Mergeloss[%3d] = %.4f\n", i, MergeLoss(i));
  }
  printf("mean = %.4f\n", merge_mean);
  printf("std  = %.4f\n", merge_std);

}/*}}}*/

int SegTree::StartT(int seg_idx) const {/*{{{*/
  if (seg_idx < 0) {
    cerr << "SegTree::StartT(" << seg_idx << "): out of bound\n";
  }
  if (seg_idx >= num_node) {
    cerr << "SegTree::StartT(" << seg_idx << "): out of bound, num_node = "
        << num_node << "\n";
  }
  assert(seg_idx >= 0 && seg_idx < num_node);
  if (seg_idx < num_leaf) {
    return seg_idx;
  } else {
    return start_t[seg_idx - num_leaf];
  }
}/*}}}*/

int SegTree::EndT(int seg_idx) const {/*{{{*/
  if (seg_idx < 0) {
    cerr << "SegTree::StartT(" << seg_idx << "): out of bound\n";
  }
  if (seg_idx >= num_node) {
    cerr << "SegTree::StartT(" << seg_idx << "): out of bound, num_node = "
        << num_node << "\n";
  }
  assert(seg_idx >= 0 && seg_idx < num_node);
  if (seg_idx < num_leaf) {
    return seg_idx;
  } else {
    return end_t[seg_idx - num_leaf];
  }
}/*}}}*/

int SegTree::Parent(int child_id) const {/*{{{*/
  assert(child_id >= 0 && child_id < num_node);
  return (child_id != num_node - 1) ? parent[child_id] : -1;
}/*}}}*/

int SegTree::Child(int parent_id, int which_child) const {/*{{{*/
  assert(parent_id >= 0 && parent_id < num_node);
  return (parent_id < num_leaf) ? -1 :
    child[2 * (parent_id - num_leaf) + which_child];
}/*}}}*/

float SegTree::MergeLoss(int seg_idx) const {/*{{{*/
  assert(seg_idx >= 0 && seg_idx < num_node);
  return (seg_idx < num_leaf) ? 0.0 : merge_loss[seg_idx - num_leaf];
}/*}}}*/

void SegTree::GetBasicSeg(vector<int>* index, float threshold) {/*{{{*/

  /* Do DFS on segtree (BFS mess up the order) */
  index->clear();
  std::list<int> nodes_to_visit;
  nodes_to_visit.push_front(num_node - 1);

  while (!nodes_to_visit.empty()) {
    int visiting = nodes_to_visit.front();
    nodes_to_visit.pop_front();
    if (MergeLoss(visiting) <= threshold) {
      index->push_back(visiting);
    } else {
      nodes_to_visit.push_front(Child(visiting, 1));
      nodes_to_visit.push_front(Child(visiting, 0));
    }
  }

}/*}}}*/

void SegTree::GetBasicSeg(vector<int>* index, int nseg) {

  assert(nseg >= 1 && nseg <= num_node);

  deque<int> seg;

  /* Do BFS on segtree until there are nseg */
  seg.push_back(num_node - 1);
  while (static_cast<int>(seg.size()) < nseg) {
    deque<int>::iterator itr = max_element(seg.begin(), seg.end());
    int first_child = Child(*itr, 0);
    int second_child = Child(*itr, 1);
    *itr = second_child;
    seg.insert(itr, first_child);
  }

  index->resize(nseg);
  copy(seg.begin(), seg.end(), index->begin());

}



