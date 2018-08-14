#ifndef __THREAD_UTIL_H__
#define __THREAD_UTIL_H__

#include <vector>
#include <iostream>
#include <string>
using std::cout;
using std::vector;
using std::string;

template<class _Tp>
class Dispatcher {/*{{{*/
  public:
    Dispatcher () {/*{{{*/
      Clear();
      pthread_mutexattr_t attr;
      pthread_mutexattr_init(&attr);
      pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_FAST_NP);
      pthread_mutex_init(&f_mutex, &attr);
      verbose = false;
      verbose_interval = 1;
    }/*}}}*/
    void Clear(){/*{{{*/
      objects.clear();
      ctr = 0;
    }/*}}}*/
    _Tp* GetObjPtr(const bool lock = true){/*{{{*/
      _Tp* objptr = NULL;

      if (lock)
        pthread_mutex_lock(&f_mutex);

      if (ctr < objects.size()) {
        objptr = &objects[ctr];
        ctr++;
      }

      do {
        if (objects.empty()) break;
        if (!verbose) break;
        if (objptr != NULL && ctr % verbose_interval != 0) break;
        cout << "\r" << 100 * ctr / objects.size() << "%  ";
        cout.flush();
      } while (false);

      if (lock)
        pthread_mutex_unlock(&f_mutex);

      return objptr;
    }/*}}}*/
    void Push(const _Tp &obj) { objects.push_back(obj); }
    void Reset() { ctr = 0; }
    void Verbose() { verbose = true; }
    void Quiet() { verbose = false; }
    void SetVerboseInt(unsigned n) { verbose_interval = n > 1 ? n : 1; }
    unsigned count() const { return ctr + 1; }
    unsigned size() const { return objects.size(); }
    const _Tp& operator[](unsigned i) const { return objects[i]; }

  private:
    vector<_Tp> objects;
    unsigned ctr;
    pthread_mutex_t f_mutex;
    bool verbose;
    unsigned verbose_interval;
};/*}}}*/

class ThreadRunner {/*{{{*/
  public:
    ThreadRunner() {}
    virtual ~ThreadRunner() {}
    virtual void* Run() = 0;
};/*}}}*/

void* ThreadEntry(void* arg);

template<class _Tp>
void CastThreads(vector<_Tp>& runner) {/*{{{*/
  if (runner.empty()) return;
  vector<pthread_t> threads(runner.size() - 1);
  for (unsigned t = 0; t < threads.size(); ++t) {
    pthread_create(&threads[t], NULL, ThreadEntry, &runner[t]);
  }
  runner.back().Run();
  void* status;
  for (unsigned t = 0; t < threads.size(); ++t) {
    pthread_join(threads[t], &status);
  }
}/*}}}*/

typedef std::pair<unsigned, unsigned> UPair;

inline void PrintDispatcherUPair(const Dispatcher<UPair>& dp) {/*{{{*/
  for (unsigned i = 0; i < dp.size(); ++i) {
    cout << "(" << dp[i].first << ", " << dp[i].second << ")\n";
  }
}/*}}}*/

#endif
