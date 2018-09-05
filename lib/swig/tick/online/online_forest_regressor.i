// License: BSD 3 clause

%include std_shared_ptr.i
%shared_ptr(OnlineForestRegressor);

%{
#include "tick/online/online_forest_regressor.h"
%}


enum class CriterionRegressor {
  unif = 0,
  mse
};

class OnlineForestRegressor {
 public:
  OnlineForestRegressor(uint32_t n_trees, double step, CriterionRegressor criterion,
                        int32_t n_threads, int seed, bool verbose);

  void fit(const SArrayDouble2dPtr features, const SArrayDoublePtr labels);
  void predict(const SArrayDouble2dPtr features, SArrayDoublePtr predictions, bool use_aggregation);

  inline double step() const;
  void print();

  ulong n_samples() const;
  ulong n_features() const;
  OnlineForestRegressor &n_features(ulong n_features);

  // uint32_t n_trees() const;
  // OnlineForestRegressor &n_trees(uint32_t n_trees);

  int32_t n_threads() const;
  OnlineForestRegressor &n_threads(int32_t n_threads);
  CriterionRegressor criterion() const;
  OnlineForestRegressor &criterion(CriterionRegressor criterion);
  int seed() const;
  OnlineForestRegressor &seed(int seed);
  // bool verbose() const;
  // OnlineForestRegressor &verbose(bool verbose);
};
