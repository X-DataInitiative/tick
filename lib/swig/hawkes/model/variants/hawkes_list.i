// License: BSD 3 clause


%{
#include "tick/hawkes/model/base/hawkes_list.h"
%}

class ModelHawkesList : public Model {

 public:
  ModelHawkesList(const int max_n_threads = 1,
                  const unsigned int optimization_level = 0);

  void set_data(const SArrayDoublePtrList2D &timestamps_list, const VArrayDoublePtr end_time);

  VArrayDoublePtr get_end_times() const;
  ulong get_n_coeffs() const;
  ulong get_n_threads() const;
  ulong get_n_nodes() const;
  ulong get_n_total_jumps() const;
  SArrayULongPtr get_n_jumps_per_node() const;
  SArrayULongPtr get_n_jumps_per_realization() const;

  void set_n_threads(const int max_n_threads);
};
