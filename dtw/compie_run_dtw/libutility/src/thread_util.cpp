#include "thread_util.h"

void* ThreadEntry(void* arg) {
  ThreadRunner *objptr = static_cast<ThreadRunner *>(arg);
  return objptr->Run();
}
