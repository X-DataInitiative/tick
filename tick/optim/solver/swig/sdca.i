%include <std_shared_ptr.i>

%{
#include "sdca.h"
#include "model.h"
%}

class SDCA : public StoSolver {

public:

    SDCA(double l_l2sq,
         unsigned long epoch_size = 0,
         double tol = 0.,
         RandType rand_type = RandType::unif,
         int seed = -1);

    void set_model(std::shared_ptr<Model> model);

    void set_prox(std::shared_ptr<Prox> prox);

    void reset();

    void init_stored_variables();

    void solve();

    double get_l_l2sq() const;

    void set_l_l2sq(double l_l2sq);

};
