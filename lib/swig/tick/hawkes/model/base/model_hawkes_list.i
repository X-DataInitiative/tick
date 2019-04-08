// License: BSD 3 clause


%{
#include "tick/hawkes/model/base/model_hawkes_list.h"
%}

class ModelHawkesList : public ModelHawkes {

 public:
  ModelHawkesList(const int max_n_threads = 1,
                  const unsigned int optimization_level = 0);

  void set_data(const SArrayDoublePtrList2D &timestamps_list, const VArrayDoublePtr end_time);

  VArrayDoublePtr get_end_times() const;
  ulong get_n_coeffs() const;
  ulong get_n_threads() const;
  SArrayULongPtr get_n_jumps_per_realization() const;

  SArrayDoublePtrList2D get_timestamps_list() const;

  void set_n_threads(const int max_n_threads);
};
