#include <cstring>
#include <typeinfo>
#include <set>
#include <cmath>
#include <algorithm>
#include "utility.h"
#include "std_common.h"
#include "thread_util.h"


using std::cout;
using std::cerr;
using std::setprecision;

namespace StdCommonUtil {

int GetDidx(const vector<string>& doc_list, string name) {/*{{{*/
  size_t found;
  unsigned didx;
  for (didx = 0; didx < doc_list.size(); ++didx) {
    if ((found = doc_list[didx].find(name)) != string::npos) break;
  }
  if (didx < doc_list.size())
    return didx;
  else
    return -1;
}/*}}}*/



void ParseList(const char *filename,/*{{{*/
               vector<string> *list,
               QueryProfileList *profile_list) {

  /* Parse filename -> (profile_list and) list */
  FILE *fd = FOPEN(filename, "r");
  char buff[1024];

  while (fgets(buff, 1024, fd)) {

    int idx = -1;
    char *tok = strtok(buff," \t\n");

    // Ignore lines starts with #
    if (tok == NULL || *tok == '#') continue;

    // First item is qid
    if (profile_list != NULL) {
      int qid = atoi(tok);
      if ((idx = profile_list->Find(qid)) != -1) {
        fprintf(stderr, "Warning %s: Repeated query id %d for query"
                "instance, overwrite old one(s)\n",
                __FUNCTION__, qid);
      } else {
        profile_list->push_back(qid);
      }
      tok = strtok(NULL," \t\n");
    }

    if (idx == -1) {
      list->push_back(tok);
    } else {
      (*list)[idx] = tok;
    }
  }
  fclose(fd);
}/*}}}*/

void ParseIgnore(const char* filename,/*{{{*/
                 vector<string>& doc_list,
                 QueryProfileList* profile_list) {
  FILE *fd = FOPEN(filename, "r");
  char buff[1024];
  /* Parse each line: qid doc_name */
  while (fgets(buff, 1024, fd)) {

    /* 1st field, qid */
    char *tok = strtok(buff, " \t\n");
    if (tok == NULL || *tok == '#') continue;

    int qid = atoi(tok);
    int qidx = profile_list->Find(qid);
    if (qidx == -1) { // Unknown qid
      fprintf(stderr, "Warning %s: Query id %d unknown, ignore.\n",
              __FUNCTION__, qid);
      continue;
    }
    /* 2nd field, doc_name */
    string name = strtok(NULL," \t\n");
    int didx = GetDidx(doc_list, name);
    if (didx == -1) { // Unknown didx
      fprintf(stderr, "Warning %s: Doc %s not found, ignore.\n",
              __FUNCTION__, name.c_str());
    }
    if (qidx != -1 && didx != -1) profile_list->PushBackIgnore(qidx, didx);
  }
  fclose(fd);
  /* Sort ignore list */
  profile_list->SortIgnore();
}/*}}}*/


void SnippetProfileList::Stat(float* p_mean, float* p_std,/*{{{*/
                              int begin, int end) {
  float& mean = *p_mean;
  float& std = *p_std;

  if (end < 0 || end > static_cast<int>(size())) end = size();
  if (begin < 0 || begin >= static_cast<int>(size())) begin = 0;

  mean = 0.0f;
  std = 0.0f;

  int nsnippet = 0;

  for (int s = begin; s < end; ++s) {
    float score = profiles[s].Score();
    if (score != -float_inf) {
      nsnippet++;
      mean += score;
      std += score * score;
    }
  }
  mean /= nsnippet;
  std = sqrt(std / nsnippet - mean * mean);
}/*}}}*/


void SnippetProfileList::Normalize(int begin, int end) {/*{{{*/

  if (end < 0 || end > static_cast<int>(size())) end = size();
  if (begin < 0 || begin >= static_cast<int>(size())) begin = 0;

  float mean, std;
  Stat(&mean, &std, begin, end);
  Normalize(mean, std, begin, end);
}/*}}}*/


void SnippetProfileList::Normalize(float mean, float std,/*{{{*/
                                   int begin, int end) {

  if (end < 0 || end > static_cast<int>(size())) end = size();
  if (begin < 0 || begin >= static_cast<int>(size())) begin = 0;

  for (int s = begin; s < end; ++s) {
    profiles[s].ScoreRef() -= mean;
    profiles[s].ScoreRef() /= std;
  }
}/*}}}*/


void SnippetProfileList::Add(float self_weight, float ref_weight,
                             const SnippetProfileList& list,
                             int begin, int end) {

  if (end < 0 || end > static_cast<int>(size())) end = size();
  if (begin < 0 || begin >= static_cast<int>(size())) begin = 0;

  if (static_cast<int>(list.size()) < end) {
    cerr << "SnippetProfileList::Add(): error: list size not matched\n";
    return;
  }

  for (int s = begin; s < end; ++s) {
    SnippetProfile& host = profiles[s];
    const SnippetProfile& guest = list.GetProfile(s);
    if (host.Qidx() != guest.Qidx()) {
      cerr << "SnippetProfileList::Add(): error: profile[" << s
           << "].Qidx() not matched\n";
    } else if (host.Didx() != guest.Didx()) {
      cerr << "SnippetProfileList::Add(): error: profile[" << s
           << "].Didx() not matched\n";
    } else if (host.NthSnippet() != guest.NthSnippet()) {
      cerr << "SnippetProfileList::Add(): error: profile[" << s
           << "].NthSnippet() not matched\n";
    } else {
      host.ScoreRef() = self_weight * host.Score() + ref_weight * guest.Score();
    }
  }

}/*}}}*/

void SnippetProfileList::Align(const SnippetProfileList& ref) {/*{{{*/
  for (unsigned sidx = 0; sidx < ref.profiles.size(); ++sidx) {
    vector<SnippetProfile>::iterator itr;
    itr = find(profiles.begin() + sidx, profiles.end(), ref.profiles[sidx]);
    std::swap(*itr, profiles[sidx]);
  }
}/*}}}*/

ostream& operator<<(ostream& os, const SnippetProfile& snippet) {
  os << std::right
    << setw(4) << snippet.Qidx()
    << setw(5) << snippet.Didx()
    << setw(4) << snippet.NthSnippet()
    << ' ' << setprecision(3) << snippet.Score()
    << " (" << snippet.Boundary().first
    << ", " << snippet.Boundary().second << ")";
  return os;
}

ostream& operator<<(ostream& os, const SnippetProfileList& snippet_list) {/*{{{*/
  os << "qidx didx nth score (start, end)\n";
  for (unsigned i = 0; i < snippet_list.size(); ++i) {
    os << snippet_list.GetProfile(i) << endl;
  }
  return os;
}/*}}}*/


AnswerList::AnswerList(const string& filename,/*{{{*/
                       const QueryProfileList& query_prof_list,
                       const vector<string>& doc_list) {
  Init(filename, query_prof_list, doc_list);
}/*}}}*/

void AnswerList::Init(const string& filename,/*{{{*/
                      const QueryProfileList& profile_list,
                      const vector<string>& doc_list) {
  is_answer.Resize(profile_list.size(), doc_list.size());
  is_answer.Memfill(false);


  FILE *fd = FOPEN(filename.c_str(), "r");
  char buff[1024];
  /* Parse each line: qid X doc_name X */
  while (fgets(buff, 1024, fd)) {
    /* 1st field, qid */
    char *tok = strtok(buff, " \t");
    if (tok == NULL || *tok == '#') continue;
    int qid = atoi(tok);
    int qidx = profile_list.Find(qid);
    if (qidx == -1) { // Unknown qid
      fprintf(stderr, "%s:%d: Query id %d unknown, ignore.\n",
              __FILE__, __LINE__, qid);
      continue;
    }
    /* 2nd field, not used */
    tok = strtok(NULL, " \t");
    /* 3rd field, doc_name */
    string name = strtok(NULL," \t\n");
    int didx = GetDidx(doc_list, name);
    if (didx == -1) {// Unknown didx
      fprintf(stderr, "%s:%d: Doc %s not found, ignore.\n",
              __FILE__, __LINE__, name.c_str());
      continue;
    }
    is_answer[qidx][didx] = true;
  }
  fclose(fd);

}/*}}}*/

ostream& operator<<(ostream& os, const AnswerList& ans_list) {/*{{{*/
  for (unsigned qidx = 0; qidx < ans_list.NumQ(); ++qidx) {
    for (unsigned didx = 0; didx < ans_list.NumD(); ++didx) {
      if(ans_list.IsAnswer(qidx, didx))
        os << "(qidx = " << qidx << ", didx = " << didx << ")\n";
    }
  }
  return os;
}/*}}}*/


inline void _InitDispatcherRange(Dispatcher<UPair> *disp, /*{{{*/
                                 unsigned qidx,
                                 unsigned didx_start,
                                 unsigned didx_end) {
  for (unsigned didx = didx_start; didx < didx_end; ++didx) {
    disp->Push(UPair(qidx, didx));
  }
}/*}}}*/

void InitDispatcher(Dispatcher<UPair>* disp, /*{{{*/
                    const QueryProfileList& profile_list,
                    const vector<string>& doc_list) {

  disp->Clear();

  for (unsigned qidx = 0; qidx < profile_list.size(); ++ qidx) {
    const QueryProfile& query = profile_list.QP(qidx);
    unsigned didx_start = 0;
    /* For each ignore */
    for (unsigned i = 0; i < query.ignore.size(); ++i) {
      _InitDispatcherRange(disp, qidx, didx_start, query.ignore[i]);
      didx_start = query.ignore[i] + 1;
    }
    _InitDispatcherRange(disp, qidx, didx_start, doc_list.size());
  }
}/*}}}*/

void InitDispatcher(Dispatcher<UPair>* disp,/*{{{*/
                    const vector<SnippetProfileList>& snippet_lists) {
  disp->Clear();
  for (unsigned qidx = 0; qidx < snippet_lists.size(); ++qidx) {
    const SnippetProfileList& qidx_snippets = snippet_lists[qidx];
    /*
    std::set<unsigned> docs;
    for (unsigned i = 0; i < qidx_snippets.size(); ++i) // collect docs
      docs.insert(qidx_snippets.GetProfile(i).Didx());
    for (typeof(docs.begin()) itr = docs.begin(); itr != docs.end(); ++itr)
      disp->Push(UPair(qidx, *itr));
      */
    for (unsigned sidx = 0; sidx < qidx_snippets.size(); ++sidx) {
      disp->Push(UPair(qidx, sidx));
    }
  }
}/*}}}*/


void _DumpDist(FILE* fp,/*{{{*/
               vector<vector<float> >& snippet_dist_qidx,
               int qid, unsigned didx_start, unsigned didx_end,
               const vector<string>& doc_list) {
  for (unsigned didx = didx_start; didx < didx_end; ++didx) {
    vector<float>& dist = snippet_dist_qidx[didx];
    if (dist.size() >= 1) {
      fprintf(fp,"%d %d %s %d %.6f %d\n",
              qid, 0, doc_list[didx].c_str(), 0, dist[0], didx);
    } else {
      cerr << "Warning " << __FUNCTION__
        << ": (qid, didx) = (" << qid << ", " << didx
        << ") not found.\n";
    }
  }
}/*}}}*/

#if 0
void DumpResult(const char* fname,/*{{{*/
                const QueryProfileList& profile_list,
                QDArray<vector<float> >& snippet_dist,
                const vector<string>& doc_list) {
  FILE* fp = FOPEN(fname, "w");
  /* For each query id */
  for (unsigned qidx = 0; qidx < profile_list.size(); ++qidx) {
    const QueryProfile& query = profile_list.QP(qidx);
    unsigned didx_start = 0;
    /* For each ignore */
    for (unsigned i = 0; i < query.ignore.size(); ++i) {
      _DumpDist(fp, snippet_dist[qidx], query.qid,
                didx_start, query.ignore[i], doc_list);
      didx_start = query.ignore[i] + 1;
    }
    _DumpDist(fp, snippet_dist[qidx], query.qid,
              didx_start, doc_list.size(), doc_list);
  }
  fclose(fp);
}/*}}}*/
#endif

/* If exist_doc.empty(): print every snippet
 * Else: print one document once (score = first appearance)
 */
void DumpResult(FILE* fp,/*{{{*/
                int qid,
                const SnippetProfileList& snippet_list,
                const vector<string>& doc_list,
                const AnswerList* ans_list,
                vector<bool>& exist_doc,
                const float bias) {


  for (unsigned i = 0; i < snippet_list.size(); ++i) {
    const SnippetProfile& snippet = snippet_list.GetProfile(i);
    int qidx = snippet.Qidx();
    int didx = snippet.Didx();

    if (snippet.Score() == -float_inf) continue;
    if (!exist_doc.empty() && exist_doc[didx] == true) continue;
    if (!exist_doc.empty()) exist_doc[didx] = true;

    int answer = -(i + 1);
    if (ans_list != NULL && ans_list->IsAnswer(qidx, didx)) {
      answer = -answer;
    }
    fprintf(fp,"%d %d %s %d %.6f %d\n", qid,
            snippet.Boundary().first, doc_list[didx].c_str(),
            snippet.Boundary().second, bias + snippet.Score(), answer);
  }
}/*}}}*/

#if 0
void DumpResult(string filename,/*{{{*/
                const QueryProfileList& profile_list,
                const vector<SnippetProfileList>& snippet_lists,
                const vector<string>& doc_list,
                const AnswerList* ans_list) {
  FILE* fp = FOPEN(filename.c_str(), "w");
  vector<bool> exist_doc(doc_list.size());
  /* For each query id */
  for (unsigned qidx = 0; qidx < profile_list.size(); ++qidx) {
    const QueryProfile& query = profile_list.QP(qidx);
    exist_doc.assign(exist_doc.size(), false);
    DumpResult(fp, query.qid, snippet_lists[qidx], doc_list, ans_list, exist_doc);
  }
  fclose(fp);
}/*}}}*/
#endif

void DumpResult(string filename,/*{{{*/
                const QueryProfileList& profile_list,
                const vector<const vector<SnippetProfileList>* >& v_snp_lists,
                const vector<string>& doc_list,
                const AnswerList* ans_list) {

  FILE* fp = FOPEN(filename.c_str(), "w");
  vector<bool> exist_doc(doc_list.size());

  /* For each query id */
  for (unsigned qidx = 0; qidx < profile_list.size(); ++qidx) {

      const QueryProfile& query = profile_list.QP(qidx);
      exist_doc.assign(exist_doc.size(), false);

      float bias = 0.0f;
      for (int i = v_snp_lists.size() - 1; i >= 0; --i) {

        const SnippetProfileList& snippet_list = (*v_snp_lists[i])[qidx];

        /* Not the last list, need bias */
        if (i < static_cast<int>(v_snp_lists.size() - 1)) {
          const SnippetProfileList& higher_list = (*v_snp_lists[i + 1])[qidx];
          bias += higher_list.MinScore() - snippet_list.MaxScore();
        }

        DumpResult(fp, query.qid, snippet_list, doc_list, ans_list, exist_doc,
                   bias);
      }

  }
  fclose(fp);
}/*}}}*/

void DumpSnippet(string filename,/*{{{*/
                 const QueryProfileList& profile_list,
                 const vector<SnippetProfileList>& snippet_list,
                 const vector<string>& doc_list,
                 const AnswerList* ans_list) {

  FILE* fp = FOPEN(filename.c_str(), "w");
  vector<bool> exist_doc;
  /* For each query id */
  for (unsigned qidx = 0; qidx < profile_list.size(); ++qidx) {
      const QueryProfile& query = profile_list.QP(qidx);

      DumpResult(fp, query.qid, snippet_list[qidx], doc_list, ans_list, exist_doc);

  }
  fclose(fp);
}/*}}}*/



} //namespace StdCommonUtil
