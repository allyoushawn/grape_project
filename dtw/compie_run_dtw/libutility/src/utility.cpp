#include <unistd.h>
#include <cstdarg>
#include <cstdlib>
#include <iostream>
#include <string.h>

#include "utility.h"

using std::cout;
using std::cerr;
using std::endl;
using std::right;
using std::left;
using std::fixed;
using std::setprecision;

FILE *FOPEN(const char fname[], char const flag[])
{/*{{{*/
	FILE *fd = fopen(fname, flag);
	if (fd==NULL){
		fprintf(stderr, "Unable to open file %s with flag %s\n", fname, flag);
		exit(-1);
	}
	return fd;
}/*}}}*/

char *Strcpy(char *&dest, const char src[]) {/*{{{*/
	if(dest != NULL) Free_1d_array(dest);
	dest = new char [strlen(src) + 1];
	strcpy(dest,src);
	return dest;
}/*}}}*/

void ErrorExit(const char file[], int line, int exit_value, const char *format, ...)
{/*{{{*/
	va_list args;
	char msg[1024];

	va_start(args, format);
	vsprintf(msg, format, args);
	va_end(args);
	fprintf(stderr, "Error: %s L%d: ",file,line);
	fprintf(stderr, "%s", msg);
	exit(exit_value);

}/*}}}*/

void CalCpuTime(clock_t real, struct tms *tms_start, struct tms * tms_end,
                double *rtime, double *utime, double *stime,
                bool print) {
	static long clktck = 0;
	if (clktck == 0){
		if ((clktck = sysconf(_SC_CLK_TCK)) < 0){
			fprintf(stderr,"Warning: line %d: sysconf error\n",__LINE__);
			return;
		}
	}
  *rtime = real/(double) clktck;
	*utime = (tms_end->tms_utime - tms_start->tms_utime) / (double) clktck;
	*stime = (tms_end->tms_stime - tms_start->tms_stime) / (double) clktck;
	if (print) {
		fprintf(stdout," real  %7.2f\n", *rtime);
		fprintf(stdout," user: %7.2f\n", *utime);
		fprintf(stdout," sys:  %7.2f\n", *stime);
	}
}

clock_t TimeStamp(struct tms *tms_stamp, bool *timeron) {
  clock_t real_t;
  if (*timeron == false) {
    return -1;
  } else if ((real_t = times(tms_stamp)) == -1) {
    *timeron = false;
    fprintf(stderr,"Warning: %s, line %d: times error\n",__FILE__,__LINE__);
    return -1;
  }
  return real_t;
}

void Timer::Print() {
  double rtime, utime, stime;
  double sum_rtime = 0, sum_utime = 0, sum_stime = 0;
  cout << "ID     real      user       sys      log\n";
  cout << "------------------------------------------------\n";
  for (unsigned i = 0; i < log.size(); ++i) {
    if (tic_not_toc[i]) {
      cout << "Timer is tic but not toc.";
    } else {
      CalCpuTime(e_real[i] - s_real[i],
                 &s_stamp[i], &e_stamp[i],
                 &rtime, &utime, &stime, false);
      cout.width(2 + nested_level[i]);
      cout << right << i;
      cout.width(5 - nested_level[i]);
      cout << right << ' ';
      cout.width(10);
      cout.setf(std::ios::fixed);
      cout << left << setprecision(2) << rtime;
      cout.width(10);
      cout << left << setprecision(2) << utime;
      cout.width(10);
      cout << left << setprecision(2) << stime;
      if (nested_level[i] == 0) {
        sum_rtime += rtime;
        sum_utime += utime;
        sum_stime += stime;
      }
    }
    cout << log[i] << endl;
  }
  cout << "------------------------------------------------\n";
  cout << "Sum    "
    << setw(10) << left << setprecision(2) << sum_rtime
    << setw(10) << left << setprecision(2) << sum_utime
    << setw(10) << left << setprecision(2) << sum_stime
    << "user + sys = " << setprecision(2) << sum_utime + sum_stime << "\n";
}

unsigned Timer::Tic(const char* msg) {
  unsigned i = s_stamp.size();
  log.push_back(msg);
  s_stamp.push_back(tms());
  e_stamp.push_back(tms());
  s_real.push_back(TimeStamp(&s_stamp[i], &is_working));
  e_real.push_back(0);
  tic_not_toc.push_back(true);
  nested_level.push_back(level);
  level++;
  return i;
}

void Timer::Toc(unsigned i) {
  if (!tic_not_toc[i]) {
    cerr << "Warning: timer[" << i << "] re-stamped, overwrite.\n";
  } else {
    level--;
  }
  e_real[i] = TimeStamp(&e_stamp[i], &is_working);
  tic_not_toc[i] = false;
}


void StripExtension(string* input) {/*{{{*/
  size_t pos = input->find_last_of(".");
  *input = input->substr(0, pos);
}/*}}}*/


void KeepBasename(string* input) {/*{{{*/
  size_t pos1 = input->find_last_of("/");
  size_t pos2 = input->find_last_of(".");
  *input = input->substr(pos1 + 1, pos2 - pos1 - 1);
}/*}}}*/

void KeepBasename(vector<string>* list) {/*{{{*/
  for (unsigned i = 0; i < list->size(); ++i) {
    KeepBasename(&(*list)[i]);
  }
}/*}}}*/

string GetExtension(const string& input) {
  string ext;
  int slash_pos = input.find_last_of("/");
  if (slash_pos == static_cast<int>(string::npos)) slash_pos = -1;
  int dot_pos = input.substr(slash_pos + 1).find_first_of(".");
  if (dot_pos != static_cast<int>(string::npos)) {
    ext = input.substr(slash_pos + dot_pos + 2);
  } else {
    ext.clear();
  }
  return ext;
}

vector<string> split(string str, string delim) {
  vector<string> tokens;
  size_t head = 0, pos = str.find_first_of(delim, head);
  while (pos != string::npos) {
    tokens.push_back(str.substr(head, pos - head));
    head = pos + 1;
    pos = str.find_first_of(delim, head);
  }
  if (head < str.length())
    tokens.push_back(str.substr(head, str.length() - head));
  return tokens;
}

istream& Getline(istream& is, string& str) {
  while (getline(is, str)) {
    if (!str.empty()) break;
  }
  return is;
}

