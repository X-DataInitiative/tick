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

  inline double step() const;
  void print();

  inline ulong n_samples() const;
  inline ulong n_features() const;
  inline OnlineForestRegressor &set_n_features(ulong n_features);
  inline uint32_t n_trees() const;
  inline OnlineForestRegressor &set_n_trees(uint32_t n_trees);
  inline int32_t n_threads() const;
  inline OnlineForestRegressor &set_n_threads(int32_t n_threads);
  inline Criterion criterion() const;
  inline OnlineForestRegressor &set_criterion(Criterion criterion);
  inline int seed() const;
  inline OnlineForestRegressor &set_seed(int seed);
  inline bool verbose() const;
  inline OnlineForestRegressor &set_verbose(bool verbose);
};
