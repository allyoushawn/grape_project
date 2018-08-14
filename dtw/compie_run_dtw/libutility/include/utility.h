#ifndef UTILITY_H
#define UTILITY_H

#include <sys/times.h>
#include <cassert>
#include <vector>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstring>

using std::endl;
using std::vector;
using std::string;
using std::ostream;
using std::setw;
using std::istream;

char *Strcpy(char *&dest, const char src[]);

template <class _Tp>
class mem_op {/*{{{*/
	public:
		static _Tp ** new_2d_array(int s_m, int s_n) {
			_Tp **ret;
			ret = new _Tp*[s_m];
			ret[0] = new _Tp[s_m*s_n];
			for( int i = 1; i < s_m; i++ ) ret[i] = ret[i-1] + s_n;
			return ret;
		}
		static bool delete_2d_array(_Tp ***ptr) {
			if (*ptr == NULL) return false;
			delete [] (*ptr)[0];
			delete [] *ptr;
			*ptr = NULL;
			return true;
		}
		static void reallocate_2d_array(_Tp ***ptr,
                                    const int m, const int n,
                                    int *m_size, int *n_size,
                                    int *array_size) {
			/* Not enough space */
			if (m * n > *array_size ) {
				mem_op<_Tp>::delete_2d_array(ptr);
				*ptr = mem_op<_Tp>::new_2d_array(m, n);
				*m_size = m;
				*n_size = n;
				*array_size = m * n;
			} else if (m > *m_size) {
        /* Enough space but m > *m_size */
				_Tp *space_ptr = (*ptr)[0];
				delete [] *ptr;
				*ptr = new _Tp*[m];
				(*ptr)[0] = space_ptr;
				for(int i = 1; i < m; ++i) {
          (*ptr)[i] = (*ptr)[i-1] + n;
        }
				*m_size = m;
				*n_size = n;
			} else if (n != *n_size) {
        /* Enough space, m < m_size, but n != n_size */
        *n_size = n;
				for(int i = 1; i < *m_size; ++i) {
          (*ptr)[i] = (*ptr)[i-1] + n;
        }
      }
		}
};/*}}}*/

template <class _Tp>
bool Free_1d_array(_Tp *&ptr)
{/*{{{*/
	if (ptr == NULL) return false;
	delete [] ptr;
	ptr = NULL;
	return true;
}/*}}}*/

template<class _Tp>
class TwoDimVector {/*{{{*/
  public:
    TwoDimVector() {}
    TwoDimVector(unsigned nrow, unsigned ncol) { resize(nrow, ncol); }
    void resize(unsigned nrow, unsigned ncol) {
      array.resize(nrow);
      for (unsigned q = 0; q < nrow; ++q)
        array[q].resize(ncol);
    }
    _Tp& operator()(unsigned q, unsigned d) {
      assert(q < array.size() && d < array[q].size());
      return array[q][d];
    }
    const _Tp& operator()(unsigned q, unsigned d) const {
      assert(q < array.size() && d < array[q].size());
      return array[q][d];
    }
    vector<_Tp>& operator[](unsigned q) {
      assert(q < array.size());
      return array[q];
    }
    const vector<_Tp>& operator[](unsigned q) const {
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
    unsigned nrow() const { return array.size(); };
    unsigned ncol() const {
      if (!array.empty()) return array[0].size();
      else return 0u;
    }

    typename vector<vector<_Tp> >::const_iterator begin() const {
      return array.begin();
    }
    typename vector<vector<_Tp> >::iterator begin() {
      return array.begin();
    }
    typename vector<vector<_Tp> >::const_iterator end() const {
      return array.end();
    }
    typename vector<vector<_Tp> >::iterator end() {
      return array.end();
    }
  private:
    vector<vector<_Tp> > array;
};/*}}}*/

template<class _Tp>
ostream& operator<<(ostream& os, const vector<_Tp>& op) {
  os << "{";
  for (unsigned i = 0; i < op.size(); ++i) {
    os << ' ' << op[i];
  }
  os << "}";
  return os;
}

template<class _Tp>
ostream& operator<<(ostream& os, const TwoDimVector<_Tp>& op) {
  for (unsigned r = 0; r < op.nrow(); ++r) {
    os << setw(3) << r << ":";
    for (unsigned c = 0; c < op.ncol(); ++c) {
      os /*<< " "*/ << op[r][c];
    }
    os << "\n";
  }
  return os;
}


/* Maitain a 2D array stored as an 1D array (continuous space)
 * R(): current number of rows
 * C(): current number of columns
 * Resize(nr, nc): resize to nr-by-nc array
 * Memfill(val): fill the array with val
 * (r, c): return the (r, c) entry (reference to entry)
 * [r]: return the [r] vector (pointer of (r, 0))
 */
template<class _Tp>
class TwoDimArray {/*{{{*/
  public:
    /* constructor */
    TwoDimArray() {
      Init();
    }
    /* copy constructor */
    TwoDimArray(const TwoDimArray& op) {
      Init();
      *this = op;
    }
    /* constructor with dimensions */
    TwoDimArray(int nr, int nc) {
      Init();
      Resize(nr, nc);
    }
    /* operator= */
    const TwoDimArray& operator=(const TwoDimArray& op) {
      Resize(op.R(), op.C());
      memcpy(data_, op.data_, op.R() * op.C() * sizeof(_Tp));
      return *this;
    }

    ~TwoDimArray() {
      mem_op<_Tp>::delete_2d_array(&data_);
    }
    void Resize(const int nr, const int nc) {
      nr_ = nr;
      nc_ = nc;
      mem_op<_Tp>::reallocate_2d_array(&data_, nr, nc,
                                       &nr_max_, &nc_max_, &size_);
    }
    void Memfill(const _Tp& val) {
      if (data_ != NULL) {
        for (int i = 0; i < nr_ * nc_; ++i) {
          data_[0][i] = val;
        }
      }
    }
    inline _Tp& operator()(const int r, const int c) {
      assert(data_ != NULL);
      assert(r >= 0 && r < nr_);
      assert(c >= 0 && c < nc_);
      return data_[r][c];
    }
    inline _Tp Entry(const int r, const int c) const {
      assert(data_ != NULL);
      assert(r >= 0 && r < nr_);
      assert(c >= 0 && c < nc_);
      return data_[r][c];
    }
    inline _Tp* operator[](const int r) {
      if (data_ == NULL) {
        return NULL;
      } else {
        assert(r >= 0 && r < nr_);
        return data_[r];
      }
    }
    inline const _Tp* Vec(const int r) const {
      if (data_ == NULL) {
        return NULL;
      } else {
        assert(r >= 0 && r < nr_);
        return data_[r];
      }
    }
    inline _Tp** Ptr() const { return data_; }
    inline int R() const { return nr_; }
    inline int C() const { return nc_; }
    inline int R_max() const { return nr_max_; }
    inline int C_max() const { return nc_max_; }
  private:
    void Init() {
      data_ = NULL;
      nr_ = nc_ = nr_max_ = nc_max_ = size_ = 0;
    }
    _Tp** data_;
    int nr_;
    int nc_;
    int nr_max_;
    int nc_max_;
    int size_;
};/*}}}*/

template<class _Tp>
ostream& operator<<(ostream& os, const TwoDimArray<_Tp>& tda) {/*{{{*/
  for (int r = 0; r < tda.R(); ++r) {
    os << setw(3) << r << ":";
    for (int c = 0; c < tda.C(); ++c) {
      os << " " << tda.Entry(r, c);
    }
    os << endl;
  }
  return os;
}/*}}}*/

void CalCpuTime(clock_t real, struct tms *tms_start, struct tms * tms_end,
                double *rtime, double *utime, double *stime,
                bool print);

clock_t TimeStamp(struct tms *tms_stamp, bool *timeron);

FILE *FOPEN(const char fname[], char const flag[]);

void ErrorExit(const char file[], int line, int exit_value, const char *format, ...);

class Timer {
  public:
    Timer() { is_working = true; level = 0;}
    void Print();
    unsigned Tic(const char* msg);
    void Toc(unsigned i);
    bool NotToc(unsigned i) const {
      assert(i < tic_not_toc.size());
      return tic_not_toc[i];
    }
  private:
    vector<string> log;
    vector<struct tms> s_stamp, e_stamp;
    vector<clock_t> s_real, e_real;
    vector<bool> tic_not_toc;
    vector<unsigned> nested_level;
    unsigned level;
    bool is_working;
};

inline void InstallTimer(Timer*& timer_ptr, Timer& timer_obj) {
  timer_ptr = &timer_obj;
}

void StripExtension(string* input);

void KeepBasename(string* input);

void KeepBasename(vector<string>* list);

string GetExtension(const string& input);

struct ReplaceExt {
  ReplaceExt(string e) { ext = e; }
  void operator()(string& str) {
    size_t pos = str.find_last_of('.');
    str = str.substr(0, pos + 1) + ext;
  }
  string ext;
};

vector<string> split(string str, string delim = " \t\n");

istream& Getline(istream& is, string& str);

#endif
