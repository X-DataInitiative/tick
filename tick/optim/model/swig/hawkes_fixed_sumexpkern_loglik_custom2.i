// License: BSD 3 clause

%{
#include "hawkes_fixed_sumexpkern_loglik_custom2.h"

%}

%{
#include "defs.h"
%}

%include <std_shared_ptr.i>

class ModelHawkesSumExpCustomType2 : public Model {

public:
    ModelHawkesSumExpCustomType2(const ArrayDouble &_decays, const ulong _MaxN, const int max_n_threads = 1);

    void set_data(const SArrayDoublePtrList1D &_timestamps,
                                 const SArrayLongPtr _global_n,
                                 const double _end_times);

    void compute_weights();

    inline unsigned long get_rand_max() const;

    //! we have implemented loss and grad, but not loss_and_grad
    double loss(const ArrayDouble &coeffs);

    void grad(const ArrayDouble &coeffs, ArrayDouble &out);

    SArrayDoublePtr get_decays() const;

    void set_decays(ArrayDouble decays);

    unsigned int get_n_threads() const;

    void set_n_threads(unsigned int n_threads);

    ulong get_n_total_jumps() const;

    ulong get_n_coeffs() const;

    ulong get_n_nodes() const;
};
