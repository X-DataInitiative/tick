// License: BSD 3 clause


%{
#include "variants/hawkes_fixed_sumexpkern_QRH3_leastsq_list.h"
%}

class ModelHawkesFixedSumExpKernLeastSqQRH3List : public ModelHawkesList {
public:
  ModelHawkesFixedSumExpKernLeastSqQRH3List(const ArrayDouble &decays, const ulong _MaxN, const int max_n_threads);

  ulong get_n_coeffs() const override;

  void set_data(const SArrayDoublePtrList2D &timestamps_list, const SArrayLongPtrList1D &global_n_list,
                                                              const VArrayDoublePtr end_times);

  double loss(const ArrayDouble &coeffs) override;
  void grad(const ArrayDouble &coeffs, ArrayDouble &out) override;
  double loss_and_grad(const ArrayDouble &coeffs, ArrayDouble &out);
};
