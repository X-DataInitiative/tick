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
               Criterion criterion,
               int max_depth,
               uint32_t min_samples_split,
               int32_t n_threads,
               int seed,
               bool verbose,
               bool warm_start,
               uint32_t n_splits);

  void fit(ulong n_iter = 0);

  void set_data(const SArrayDouble2dPtr features, const SArrayDoublePtr labels);

  void predict(const SArrayDouble2dPtr features, SArrayDoublePtr predictions);

  void print();

  inline ulong n_features() const;
  inline ulong n_samples() const;

  inline uint32_t n_trees() const;
  inline OnlineForest& set_n_trees(uint32_t n_trees);

  inline uint32_t n_splits() const;
  inline OnlineForest& set_n_splits(uint32_t n_splits);

  inline int32_t n_threads() const;
  inline OnlineForest& set_n_threads(int32_t n_threads);

  inline uint32_t min_samples_split() const;
  inline OnlineForest& set_min_samples_split(uint32_t min_samples_split);

  inline Criterion criterion() const;
  inline OnlineForest& set_criterion(Criterion criterion);

  inline int32_t max_depth() const;
  inline OnlineForest& set_max_depth(int32_t max_depth);

  inline int seed() const;
  inline OnlineForest& set_seed(int seed);

  inline bool verbose() const;
  inline OnlineForest& set_verbose(bool verbose);

  inline bool warm_start() const;
  inline OnlineForest& set_warm_start(bool warm_start);
};
