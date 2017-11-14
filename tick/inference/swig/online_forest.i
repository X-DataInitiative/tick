// License: BSD 3 clause

%include std_shared_ptr.i
%shared_ptr(OnlineForestRegressor);

%{
#include "online_forest.h"
%}


enum class Criterion {
  unif = 0,
  mse
};

class OnlineForestRegressor {
 public:
  OnlineForestRegressor(uint32_t n_trees, double step, Criterion criterion, int32_t n_threads, int seed, bool verbose);

  void fit(const SArrayDouble2dPtr features, const SArrayDoublePtr labels);
  void predict(const SArrayDouble2dPtr features, SArrayDoublePtr predictions, bool use_aggregation);

  void clear();

  inline double step() const;
  void print();

  ulong n_samples() const;
  ulong n_features() const;

  uint32_t n_trees() const;
  OnlineForestRegressor &set_n_trees(uint32_t n_trees);

  int32_t n_threads() const;
  OnlineForestRegressor &set_n_threads(int32_t n_threads);
  Criterion criterion() const;
  OnlineForestRegressor &set_criterion(Criterion criterion);
  int seed() const;
  OnlineForestRegressor &set_seed(int seed);

};
