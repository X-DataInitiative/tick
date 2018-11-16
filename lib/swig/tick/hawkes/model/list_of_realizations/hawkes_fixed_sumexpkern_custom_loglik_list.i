// License: BSD 3 clause


%{
#include "tick/hawkes/model/list_of_realizations/hawkes_fixed_sumexpkern_custom_loglik_list.h"
%}

%shared_ptr(ModelHawkesFixedSumExpKernCustomLogLikList);


class ModelHawkesFixedSumExpKernCustomLogLikList : public ModelHawkesCustomLogLikList {

public:

  ModelHawkesFixedSumExpKernCustomLogLikList(const ArrayDouble &decays, const ulong _MaxN_of_f,
                                       const int max_n_threads = 1);

  void set_decays(ArrayDouble &decays);
  SArrayDoublePtr get_decays() const;
  ulong get_MaxN_of_f() const;
  void set_MaxN_of_f(ulong _MaxN_of_f);
  ulong get_n_coeffs() const;

  void set_data(const SArrayDoublePtrList2D &timestamps_list, const SArrayLongPtrList1D &global_n_list,
                                                              const VArrayDoublePtr end_times);

  double loss(const ArrayDouble &coeffs) override;
  double loss_i(const ulong i, const ArrayDouble &coeffs) override;
  void grad(const ArrayDouble &coeffs, ArrayDouble &out) override;
  void grad_i(const ulong i, const ArrayDouble &coeffs, ArrayDouble &out) override;
  double loss_and_grad(const ArrayDouble &coeffs, ArrayDouble &out);
};
