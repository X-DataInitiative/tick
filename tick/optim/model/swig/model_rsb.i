// License: BSD 3 clause

%{
#include "model_rsb.h"

%}

%{
#include "defs.h"
%}

%include <std_shared_ptr.i>

class ModelRsb : public Model {

public:

    ModelRsb(const double _decay, const ulong _MaxN, const int max_n_threads = 1);

    void set_data(const SArrayDoublePtrList1D &_timestamps,
                  const SArrayLongPtr _global_n,
                  const double _end_times);

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
