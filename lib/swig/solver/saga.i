// License: BSD 3 clause

%include "sto_solver.i"

%{
#include "tick/solver/saga.h"
#include "tick/base_model/model.h"
%}

template <class T, class K>
class BaseSAGA : public TStoSolver<T, K> {
 public:
    enum class VarianceReductionMethod {
        Last    = 1,
        Average = 2,
        Random  = 3
    };
    BaseSAGA(unsigned long epoch_size,
         K tol,
         RandType rand_type,
         K step,
         int seed,
         BaseSAGA::VarianceReductionMethod variance_reduction
         = BaseSAGA::VarianceReductionMethod::Last);
    void solve();
    void set_step(K step);
    VarianceReductionMethod get_variance_reduction();
    void set_variance_reduction(VarianceReductionMethod variance_reduction);

    void set_model(std::shared_ptr<TModel<T, K> > model) override;
};

%rename(BaseSAGADouble) BaseSAGA<double, double>;
class BaseSAGADouble : public TStoSolver<double, double> {
 public:
    enum class VarianceReductionMethod {
        Last    = 1,
        Average = 2,
        Random  = 3
    };
    BaseSAGADouble(unsigned long epoch_size,
         double tol,
         RandType rand_type,
         double step,
         int seed,
         VarianceReductionMethod variance_reduction
         = VarianceReductionMethod::Last);
    void solve();
    void set_step(double step);
    VarianceReductionMethod get_variance_reduction();
    void set_variance_reduction(VarianceReductionMethod variance_reduction);

    void set_model(ModelDoublePtr model) override;
};

%rename(BaseSAGAFloat) BaseSAGA<float, float>;
class BaseSAGA<float, float> : public TStoSolver<float, float> {
 public:
    enum class VarianceReductionMethod {
        Last    = 1,
        Average = 2,
        Random  = 3
    };
    BaseSAGAFloat(unsigned long epoch_size,
         float tol,
         RandType rand_type,
         float step,
         int seed,
         VarianceReductionMethod variance_reduction
         = VarianceReductionMethod::Last);
    void solve();
    void set_step(float step);
    VarianceReductionMethod get_variance_reduction();
    void set_variance_reduction(VarianceReductionMethod variance_reduction);

    void set_model(ModelFloatPtr model) override;
};

class SAGA : public BaseSAGADouble {
public:
    SAGA(unsigned long epoch_size,
         double tol,
         RandType rand_type,
         double step,
         int seed = -1,
         BaseSAGADouble::VarianceReductionMethod variance_reduction = BaseSAGADouble::VarianceReductionMethod::Last);
    void solve();
    void set_step(double step);
    BaseSAGADouble::VarianceReductionMethod get_variance_reduction();
    void set_variance_reduction(BaseSAGADouble::VarianceReductionMethod variance_reduction);
    
    virtual void set_starting_iterate(ArrayDouble &new_iterate);
    virtual void get_iterate(ArrayDouble &out);
    virtual void get_minimizer(ArrayDouble &out);
};
