// License: BSD 3 clause

%include <std_shared_ptr.i>

%{
#include "tick/solver/sdca.h"
#include "tick/base_model/model.h"
%}

template <class T, class K = T>
class TSDCA : public TStoSolver<T, K> {
public:
    TSDCA();
    TSDCA(T l_l2sq,
         unsigned long epoch_size = 0,
         T tol = 0.,
         RandType rand_type = RandType::unif,
         int record_every = 1,
         int seed = -1);

    void set_model(std::shared_ptr<TModel<T, K>> model);
    void reset();
    void set_starting_iterate();
    void set_starting_iterate(Array<K> &dual_vector);
    T get_l_l2sq() const;
    void set_l_l2sq(T l_l2sq);
    std::shared_ptr<Array<K> > get_primal_vector();
    std::shared_ptr<Array<K> > get_dual_vector();

    bool compare(const TSDCA<T, K> &that);
};

%rename(SDCADouble) TSDCA<double>;
class SDCADouble : public StoSolverDouble {
public:
    SDCADouble();
    SDCADouble(double l_l2sq,
         unsigned long epoch_size = 0,
         double tol = 0.,
         RandType rand_type = RandType::unif,
         int record_every = 1,
         int seed = -1);

    void set_model(ModelDoublePtr model);
    void reset();
    void set_starting_iterate();
    void set_starting_iterate(ArrayDouble &dual_vector);
    double get_l_l2sq() const;
    void set_l_l2sq(double l_l2sq);
    SArrayDoublePtr get_primal_vector();
    SArrayDoublePtr get_dual_vector();

    bool compare(const SDCADouble &that);
};
typedef TSDCA<double> SDCADouble;
TICK_MAKE_PICKLABLE(SDCADouble);

%rename(SDCAFloat) TSDCA<float>;
class SDCAFloat : public StoSolverFloat {
public:
    SDCAFloat();
    SDCAFloat(float l_l2sq,
         unsigned long epoch_size = 0,
         float tol = 0.,
         RandType rand_type = RandType::unif,
         int record_every = 1,
         int seed = -1);

    void set_model(ModelFloatPtr model);
    void reset();
    void set_starting_iterate();
    void set_starting_iterate(ArrayFloat &dual_vector);
    float get_l_l2sq() const;
    void set_l_l2sq(float l_l2sq);
    SArrayFloatPtr get_primal_vector();
    SArrayFloatPtr get_dual_vector();

    bool compare(const SDCAFloat &that);
};
typedef TSDCA<float> SDCAFloat;
TICK_MAKE_PICKLABLE(SDCAFloat);