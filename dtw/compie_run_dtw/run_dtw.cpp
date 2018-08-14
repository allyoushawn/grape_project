#include <iostream>
#include "dtw_parm.h"
#include "dtw_util.h"

using std::cout;
using DtwUtil::DtwParm;
using DtwUtil::SegDtwRunner;
using DtwUtil::FrameDtwRunner;
using DtwUtil::SlopeConDtwRunner;
using DtwUtil::DeterminePhiFn;

int main(int argc, char *argv[]) {

  string q_fname = argv[1];
  string d_fname = argv[2];
  DtwParm q_parm, d_parm;
  d_parm.LoadParm(d_fname);
  vector<float> hypo_score;
  vector<pair<int, int> > hypo_bound;

  /* Use DeterminePhiFn() for local distance function selection */
//  SlopeConDtwRunner scdtw_runner(DeterminePhiFn(GetExtension(q_fname)));
//   or you choose it your own like:
//   SlopeConDtwRunner scdtw_runner(DtwUtil::euclinorm);
//   FrameDtwRunner::nsnippet_ = 1;
//
//  /* Frame-based DTW */
//  //   DtwRunners can be reuse (SegDtwRunner too)
//  //   Everytime you call InitDtw() then DTW(), it's like brand new!
//  q_parm.LoadParm(q_fname);
//  scdtw_runner.InitDtw(&hypo_score,
//                         &hypo_bound, /* (start, end) frame */
//                         NULL, /* do not backtracking */
//                         &q_parm,
//                         &d_parm,
//                         NULL, /* full time span */
//                         NULL); /* full time span */
//  scdtw_runner.DTW();
//
//  unsigned num_hypo = hypo_score.size();
//  for (unsigned i = 0; i < num_hypo; ++i) {
//    cout << hypo_score[i] << endl;
//  }


  /* Segment-based DTW */
  hypo_score.clear();
  hypo_bound.clear();
  float bseg_ratio = 0.5;
  float superseg_ratio = 4.0;
  int gran = 3, width = 3;
  d_parm.LoadParm(d_fname, bseg_ratio, superseg_ratio, width, gran, "");
  SegDtwRunner segdtw_runner(DtwUtil::euclinorm);
  SegDtwRunner::nsnippet_ = 1;

    q_parm.LoadParm(q_fname, bseg_ratio, superseg_ratio, width, gran, "");
    //q_parm.DumpData();
    segdtw_runner.InitDtw(&hypo_score,
                          &hypo_bound, /* (start, end) basic segment */
                          NULL, /* do not backtracking */
                          &q_parm,
                          &d_parm,
                          NULL, /* full time span */
                          NULL); /* full time span */
    segdtw_runner.DTW();
    size_t num_hypo = hypo_score.size();
    for (unsigned i = 0; i < num_hypo; ++i) {
      cout << hypo_score[i]<<endl;

    }


  return 0;
}
