#ifndef __STD_COMMON_H__
#define __STD_COMMON_H__

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
#include <limits>
#include "thread_util.h"
#include "utility.h"

using std::vector;
using std::string;
using std::cout;
using std::setw;
using std::ostream;


namespace StdCommonUtil {

typedef std::pair<unsigned, unsigned> UPair;
typedef std::pair<int, int> IPair;

struct FileDirExt {
  string dir;
  string ext;
  bool operator==(const FileDirExt& ref) {
    return (dir.compare(ref.dir) == 0) && (ext.compare(ref.ext) == 0);
  }
};


#define float_inf std::numeric_limits<float>::infinity()

/* A num_query by num_doc array of _Tp
 * Implemented by vector<vector<_Tp> >
 */
template<class _Tp>
class QDArray {/*{{{*/
  public:
    QDArray() {}
    QDArray(unsigned nq, unsigned nd) { Resize(nq, nd); }
    void Resize(unsigned nq, unsigned nd) {
      array.resize(nq);
      for (unsigned q = 0; q < nq; ++q)
        array[q].resize(nd);
    }
    _Tp& operator()(unsigned q, unsigned d) {
      assert(q < array.size() && d < array[q].size());
      return array[q][d];
    }
    vector<_Tp>& operator[](unsigned q) {
      assert(q < array.size());
      return array[q];
    }
    void memfill(const _Tp& val) {
      if (!array.empty()) {
        unsigned n = array[0].size();
        for (unsigned q = 0; q < array.size(); ++q)
          array[q].assign(n, val);
      }
    }
    unsigned NumQ() const { return array.size(); };
    unsigned NumD() const {
      if (!array.empty()) return array[0].size();
      else return 0u;
    }
  private:
    vector<vector<_Tp> > array;
};/*}}}*/



struct QueryProfile {/*{{{*/
  int qid;
  vector<unsigned> ignore;
};/*}}}*/

class QueryProfileList {/*{{{*/
  public:
    int Find(int qid) const {
      for (unsigned i = 0; i < profiles.size(); ++i) {
        if (profiles[i].qid == qid) return i;
      }
      return -1;
    }
    int push_back(int qid) {
      profiles.resize(profiles.size() + 1);
      profiles.back().qid = qid;
      profiles.back().ignore.clear();
      return profiles.size() - 1;
    }
    void PushBackIgnore(unsigned qidx, unsigned didx) {
      assert(qidx < profiles.size());
      profiles[qidx].ignore.push_back(didx);
    }
    unsigned size() const { return profiles.size(); }
    const QueryProfile& QP(unsigned qidx) const {
      assert(qidx < profiles.size());
      return profiles[qidx];
    }
    void SortIgnore() {
      for (unsigned i = 0; i < profiles.size(); ++i) {
        sort(profiles[i].ignore.begin(), profiles[i].ignore.end());
      }
    }
    void Print() const {
      cout << " qid  ignore_list\n";
      for (unsigned qidx = 0; qidx < profiles.size(); ++qidx) {
         cout << setw(4) << profiles[qidx].qid << "  {";
         for (unsigned j = 0; j < profiles[qidx].ignore.size(); ++j) {
           cout << " " << profiles[qidx].ignore[j];
         }
         cout << "}\n";
      }
    }
  private:
    vector<QueryProfile> profiles;
};/*}}}*/



class SnippetProfile {/*{{{*/
  public:
    SnippetProfile()
      : qidx(-1), didx(-1), nth_snippet(-1),
      score(float_inf), boundary(IPair(-1, -1)) {}
    SnippetProfile(int q, int d, int n, float s, const IPair& b) {
      Init(q, d, n, s, b);
    }
    void Init(int q, int d, int n, float s, const IPair& b) {
      qidx = q;
      didx = d;
      nth_snippet = n;
      score = s;
      boundary = b;
    }
    float& ScoreRef() { return score; }
    IPair& BoundaryRef() { return boundary; }
    int Qidx() const { return qidx; }
    int Didx() const { return didx; }
    int NthSnippet() const { return nth_snippet; }
    float Score() const { return score; }
    const IPair& Boundary() const { return boundary; }
    const int Len() const { return boundary.second - boundary.first + 1; }
  protected:
    int qidx;
    int didx;
    int nth_snippet;
    float score;
    IPair boundary;
};/*}}}*/

inline bool CompareSnippetScore(const SnippetProfile& a,/*{{{*/
                                const SnippetProfile& b) {
  return a.Score() > b.Score();
}/*}}}*/

inline bool operator==(const SnippetProfile&a, const SnippetProfile& b) {
  return a.Qidx() == b.Qidx() && a.Didx() == b.Didx() &&
    a.NthSnippet() == b.NthSnippet();
}

class SnippetProfileList {/*{{{*/
  public:
    void Resize(int n) {
      assert(n <= static_cast<int>(profiles.size()));
      assert(-n <= static_cast<int>(profiles.size()));
      if (n >= 0) {
        profiles.resize(n);
      } else {
        profiles.resize(profiles.size() + n);
      }
    }
    void Clear() { profiles.clear(); }
    unsigned size() const { return profiles.size(); }
    const SnippetProfile& GetProfile(unsigned idx) const {
      assert(idx < profiles.size());
      return profiles[idx];
    }
    SnippetProfile& ProfileRef(int idx) {
      assert(idx < static_cast<int>(profiles.size()));
      assert(-idx <= static_cast<int>(profiles.size()));
      return idx >= 0 ? profiles[idx] : profiles[profiles.size() + idx];
    }
    const SnippetProfile& Front() const {
      return profiles.front();
    }
    const SnippetProfile& Back() const {
      return profiles.back();
    }
    int push_back(int q, int d, int n, float s, const IPair& b) {
      profiles.push_back(SnippetProfile(q, d, n, s, b));
      return profiles.size() - 1;
    }
    int push_back(const SnippetProfile& sp) {
      profiles.push_back(sp);
      return profiles.size() - 1;
    }
    void Sort(unsigned i, unsigned j) {
      assert(i <= j);
      assert(j <= size());
      sort(profiles.begin() + i, profiles.begin() + j, CompareSnippetScore);
    }
    void Sort() {
      sort(profiles.begin(), profiles.end(), CompareSnippetScore);
    }
    void Stat(float* p_mean, float* p_std,
              int begin = 0, int end = -1);

    /* Normalize snippets[begin - end] with zero mean and unit std */
    void Normalize(int begin = 0, int end = -1);

    /* Normalize snippets[begin - end] assuming mean and std given */
    void Normalize(float mean, float std,
                   int begin = 0, int end = -1);
    /* Add */
    void Add(float self_weight, float ref_weight,
             const SnippetProfileList& list, int begin = 0, int end = -1);


    /* Align with ref according to qidx, didx and sidx */
    void Align(const SnippetProfileList& ref);


    float MinScore() const {
      float m = float_inf;
      for (unsigned s = 0; s < profiles.size(); ++s) {
        float score = profiles[s].Score();
        if (score != -float_inf && score < m) m = score;
      }
      if (m == float_inf) m = 0.0;
      return m;
    }

    float MaxScore() const {
      float M = -float_inf;
      for (unsigned s = 0; s < profiles.size(); ++s) {
        float score = profiles[s].Score();
        if (score != -float_inf && score > M) M = score;
      }
      if (M == -float_inf) M = 0.0;
      return M;
    }

  private:
    vector<SnippetProfile> profiles;
};/*}}}*/

ostream& operator<<(ostream& os, const SnippetProfile& snippet);

ostream& operator<<(ostream& os, const SnippetProfileList& snippet_list);


class AnswerList {/*{{{*/
  public:
    AnswerList() {}
    AnswerList(const string& filename,
               const QueryProfileList& profile_list,
               const vector<string>& doc_list);
    void Init(const string& filename,
              const QueryProfileList& query_prof_list,
              const vector<string>& doc_list);
    bool IsAnswer(int qidx, int didx) const {
      if (qidx < 0 || qidx >= static_cast<int>(is_answer.R()))
        return false;
      else if (didx < 0 || didx >= static_cast<int>(is_answer.C()))
        return false;
      else
        return is_answer.Entry(qidx, didx);
    }
    unsigned NumQ() const { return is_answer.R(); };
    unsigned NumD() const { return is_answer.C(); };
  private:
    TwoDimArray<bool> is_answer;

};/*}}}*/

ostream& operator<<(ostream& os, const AnswerList& ans_list);



void ParseList(const char *filename,
               vector<string> *list,
               QueryProfileList *profile_list = NULL);

void ParseIgnore(const char* filename,
                 vector<string>& doc_list,
                 QueryProfileList* profile_list);

void InitDispatcher(Dispatcher<UPair>* disp,
                    const QueryProfileList& profile_list,
                    const vector<string>& doc_list);

void InitDispatcher(Dispatcher<UPair>* disp,
                    const vector<SnippetProfileList>& snippet_lists);

void DumpResult(const char* fname,
                const QueryProfileList& profile_list,
                QDArray<vector<float> >& snippet_dist,
                const vector<string>& doc_list);

/* Dump list of qid */
void DumpResult(FILE* fp,
                int qid,
                const SnippetProfileList& snippet_list,
                const vector<string>& doc_list,
                const AnswerList* ans_list,
                vector<bool>& exist_doc,
                float bias = 0.0f);

/* Dump list of multiple qid's (per doc basis) */
void DumpResult(string filename,
                const QueryProfileList& profile_list,
                const vector<SnippetProfileList>& snippet_lists,
                const vector<string>& doc_list,
                const AnswerList* ans_list);

/* Dump list of multiple qid's (per doc basis), with backoff (in v_snp_lists) */
void DumpResult(string filename,
                const QueryProfileList& profile_list,
                const vector<const vector<SnippetProfileList>* >& v_snp_lists,
                const vector<string>& doc_list,
                const AnswerList* ans_list);

/* Dump list of multiple qid's (per snippet basis) */
void DumpSnippet(string filename,
                 const QueryProfileList& profile_list,
                 const vector<SnippetProfileList>& snippet_list,
                 const vector<string>& doc_list,
                 const AnswerList* ans_list);

} //namespace StdCommonUtil
#endif
