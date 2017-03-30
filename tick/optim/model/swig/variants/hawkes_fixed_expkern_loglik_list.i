
%{
#include "variants/hawkes_fixed_expkern_loglik_list.h"
%}


class ModelHawkesFixedExpKernLogLikList : public ModelHawkesList {
    
public:
    
  ModelHawkesFixedExpKernLogLikList(const double decay,
                                    const int max_n_threads = 1);

  double loss_and_grad(const ArrayDouble &coeffs, ArrayDouble &out);
  double hessian_norm(const ArrayDouble &coeffs, const ArrayDouble &vector);

  void set_decay(const double decay);

  void incremental_set_data(const SArrayDoublePtrList1D &timestamps, double end_time);

  void compute_weights();
};
