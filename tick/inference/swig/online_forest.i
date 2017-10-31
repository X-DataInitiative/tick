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


enum class Criterion {
  unif = 0,
  mse
};

class OnlineForest {
 public:
  OnlineForest(uint32_t n_trees,
               uint32_t n_min_samples,
               uint8_t n_splits);

  void fit(ulong n_iter = 0);

  void set_data(const SArrayDouble2dPtr features, const SArrayDoublePtr labels);

  void predict(const SArrayDouble2dPtr features, SArrayDoublePtr predictions);

  void print();

  uint32_t n_trees();

  uint32_t n_threads();


};
