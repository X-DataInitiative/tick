// License: BSD 3 clause

%include std_shared_ptr.i
%shared_ptr(OnlineForest);

%{
#include "online_forest.h"
%}

enum class CycleType {
  uniform = 0,
  permutation,
  sequential
};

class OnlineForest {
 public:
  OnlineForest(uint32_t n_trees, uint32_t n_min_samples, uint8_t n_splits, CycleType cycle_type);

  void fit(ulong n_iter = 0);

  void set_data(const SBaseArrayDouble2dPtr features, const SArrayDoublePtr labels);
};
