// License: BSD 3 clause

%include <std_shared_ptr.i>

%{
#include "tick/solver/sdca.h"
#include "tick/base_model/model.h"
%}

template <class T>
class TSDCA : public TStoSolver<T> {
public:
    TSDCA(T l_l2sq,
         unsigned long epoch_size = 0,
         T tol = 0.,
         RandType rand_type = RandType::unif,
         int seed = -1);

    void set_model(std::shared_ptr<TModel<T>> model);
    void reset();
    void set_starting_iterate();
    void set_starting_iterate(Array<T> &dual_vector);
    void solve();
    T get_l_l2sq() const;
    void set_l_l2sq(T l_l2sq);
    std::shared_ptr<Array<T> > get_primal_vector();
    std::shared_ptr<Array<T> > get_dual_vector();
};

%rename(SDCA) TSDCA<double>;
class TSDCA<double> : public TStoSolver<double> {
public:
    SDCA(double l_l2sq,
         unsigned long epoch_size = 0,
         double tol = 0.,
         RandType rand_type = RandType::unif,
         int seed = -1);

    void set_model(ModelDoublePtr model);
    void reset();
    void set_starting_iterate();
    void set_starting_iterate(ArrayDouble &dual_vector);
    void solve();
    double get_l_l2sq() const;
    void set_l_l2sq(double l_l2sq);
    SArrayDoublePtr get_primal_vector();
    SArrayDoublePtr get_dual_vector();
};
typedef TSDCA<double> SDCA;

%rename(SDCADouble) TSDCA<double>;
class TSDCA<double> : public TStoSolver<double> {
public:
    SDCADouble(double l_l2sq,
         unsigned long epoch_size = 0,
         double tol = 0.,
         RandType rand_type = RandType::unif,
         int seed = -1);

    void set_model(ModelDoublePtr model);
    void reset();
    void set_starting_iterate();
    void set_starting_iterate(ArrayDouble &dual_vector);
    void solve();
    double get_l_l2sq() const;
    void set_l_l2sq(double l_l2sq);
    SArrayDoublePtr get_primal_vector();
    SArrayDoublePtr get_dual_vector();
};
typedef TSDCA<double> SDCADouble;

%rename(SDCAFloat) TSDCA<float>;
class TSDCA<float> : public TStoSolver<float> {
public:
    SDCAFloat(float l_l2sq,
         unsigned long epoch_size = 0,
         float tol = 0.,
         RandType rand_type = RandType::unif,
         int seed = -1);

    void set_model(ModelFloatPtr model);
    void reset();
    void set_starting_iterate();
    void set_starting_iterate(ArrayFloat &dual_vector);
    void solve();
    float get_l_l2sq() const;
    void set_l_l2sq(float l_l2sq);
    SArrayFloatPtr get_primal_vector();
    SArrayFloatPtr get_dual_vector();
};
typedef TSDCA<float> SDCAFloat;
