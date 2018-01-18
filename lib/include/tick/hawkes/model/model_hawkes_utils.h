
#ifndef LIB_INCLUDE_TICK_HAWKES_MODEL_MODEL_HAWKES_UTILS_H_
#define LIB_INCLUDE_TICK_HAWKES_MODEL_MODEL_HAWKES_UTILS_H_

// License: BSD 3 clause

#include "tick/base/base.h"

struct TimestampListDescriptor {
  ulong n_realizations;
  ulong n_nodes;
  VArrayULongPtr n_jumps_per_realization;
  SArrayULongPtr n_jumps_per_node;
};

TimestampListDescriptor describe_timestamps_list(const SArrayDoublePtrList2D &timestamps_list);

TimestampListDescriptor describe_timestamps_list(const SArrayDoublePtrList2D &timestamps_list,
                                                 const VArrayDoublePtr end_times);

#endif  // LIB_INCLUDE_TICK_HAWKES_MODEL_MODEL_HAWKES_UTILS_H_
