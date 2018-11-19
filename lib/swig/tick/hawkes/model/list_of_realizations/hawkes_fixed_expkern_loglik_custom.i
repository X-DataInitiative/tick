// License: BSD 3 clause

%{
#include "tick/hawkes/model/list_of_realizations/hawkes_fixed_expkern_loglik_custom.h"

%}

%{
#include "tick/base/defs.h"
%}

%include <std_shared_ptr.i>
%include "tick/hawkes/model/base/model_hawkes_loglik_single.i"

class ModelHawkesCustom : public ModelHawkesLogLikSingle {

public:

    ModelHawkesCustom(const double _decay,
                      const unsigned long _MaxN_of_f, const unsigned int n_cores = 1);

    void set_data(const SArrayDoublePtrList1D &_timestamps,
                                 const SArrayLongPtr _global_n,
                                 const double _end_times);

    void compute_weights();

    inline unsigned long get_rand_max() const;

    //! we have implemented loss and grad, but not loss_and_grad
    double loss(const ArrayDouble &coeffs);

    void grad(const ArrayDouble &coeffs, ArrayDouble &out);

    double get_decay() const;

    void set_decay(double decay);

    unsigned int get_n_threads() const;

    void set_n_threads(unsigned int n_threads);

    ulong get_n_total_jumps() const;

    ulong get_n_coeffs() const;

    ulong get_n_nodes() const;
};
