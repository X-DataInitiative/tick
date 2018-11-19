// License: BSD 3 clause


%{
#include "tick/hawkes/model/base/model_hawkes_single.h"
%}


class ModelHawkesSingle : public ModelHawkes {

public:

  ModelHawkesSingle(const int max_n_threads = 1);

};



class ModelHawkesLogLikSingle : public ModelHawkesSingle {

public:

  ModelHawkesLogLikSingle(const int max_n_threads = 1);

  double loss_and_grad(const ArrayDouble &coeffs, ArrayDouble &out);
  double hessian_norm(const ArrayDouble &coeffs, const ArrayDouble &vector);
  void hessian(const ArrayDouble &coeffs, ArrayDouble &out);

  void compute_weights();
};
