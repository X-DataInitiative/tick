// License: BSD 3 clause


%{
#include "hawkes_fixed_sumexpkern_leastsq_qrh1.h"
%}

%{
#include "defs.h"
%}

%include <std_shared_ptr.i>

class ModelHawkesFixedSumExpKernLeastSqQRH1 : public Model {

public:

    ModelHawkesFixedSumExpKernLeastSqQRH1(const ArrayDouble &decays,
                                          const ulong MaxN,
                                          const unsigned int max_n_threads = 1,
                                          const unsigned int optimization_level = 0);

    void set_data(const SArrayDoublePtrList1D &_timestamps,
                  const SArrayLongPtr _global_n,
                  const double _end_times);

    double loss(const ArrayDouble &coeffs);

    void grad(const ArrayDouble &coeffs, ArrayDouble &out);

    ulong get_n_total_jumps() const;
    ulong get_n_coeffs() const;
    ulong get_n_nodes() const;


};