// License: BSD 3 clause


%{
#include "tick/hawkes/model/base/model_hawkes.h"
%}

class ModelHawkes : public Model {

 public:
  ModelHawkes(const int max_n_threads = 1,
              const unsigned int optimization_level = 0);

  void set_n_threads(const int max_n_threads);

  ulong get_n_nodes() const;
  ulong get_n_total_jumps() const;
  SArrayULongPtr get_n_jumps_per_node() const;
};
