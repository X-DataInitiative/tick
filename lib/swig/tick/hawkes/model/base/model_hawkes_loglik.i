// License: BSD 3 clause


%{
#include "tick/hawkes/model/base/model_hawkes_loglik.h"
%}


class ModelHawkesLogLik : public ModelHawkesList {

public:

  ModelHawkesLogLik(const int max_n_threads = 1);

  double loss_and_grad(const ArrayDouble &coeffs, ArrayDouble &out);
  double hessian_norm(const ArrayDouble &coeffs, const ArrayDouble &vector);
  void hessian(const ArrayDouble &coeffs, ArrayDouble &out);

  void incremental_set_data(const SArrayDoublePtrList1D &timestamps, double end_time);

  void compute_weights();
};
