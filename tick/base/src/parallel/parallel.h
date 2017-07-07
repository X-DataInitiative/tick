#ifndef TICK_BASE_SRC_PARALLEL_PARALLEL_H_
#define TICK_BASE_SRC_PARALLEL_PARALLEL_H_

#if defined(_OPENMP) && defined(TICK_USE_OPENMP)
#pragma message("Using experimental OpenMP implementation")
#include "parallel_omp.h"
#else
#include "parallel_stdthread.h"
#endif

#endif  // TICK_BASE_SRC_PARALLEL_PARALLEL_H_
