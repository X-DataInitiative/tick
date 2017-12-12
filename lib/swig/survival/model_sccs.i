// License: BSD 3 clause

%include <std_shared_ptr.i>
%shared_ptr(ModelSCCS);

%{
#include "tick/survival/model_sccs.h"
%}


class ModelSCCS : public Model {

 public:
  ModelSCCS(const SBaseArrayDouble2dPtrList1D &features,
            const SArrayIntPtrList1D &labels,
            const SBaseArrayULongPtr censoring,
            ulong n_lags);

  double loss(ArrayDouble &coeffs);

  void grad(ArrayDouble &coeffs, ArrayDouble &out);

  void compute_lip_consts();

  unsigned long get_rand_max();

  unsigned long get_epoch_size();

  // Number of parameters to be estimated. Can differ from the number of
  // features, e.g. when including an intercept.
  unsigned long get_n_coeffs() const;

  double get_lip_max();
};