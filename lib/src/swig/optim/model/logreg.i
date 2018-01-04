// License: BSD 3 clause

%{
#include "tick/optim/model/logreg.h"
%}


class ModelLogReg : public ModelGeneralizedLinear, public ModelLipschitz {
 public:
 // Empty constructor only used for serialization
   ModelLogReg(){};

  ModelLogReg(const SBaseArrayDouble2dPtr features,
              const SArrayDoublePtr labels,
              const bool fit_intercept,
              const int n_threads);

  static void sigmoid(const ArrayDouble &x, ArrayDouble &out);

  static void logistic(const ArrayDouble &x, ArrayDouble &out);
};

TICK_MAKE_PICKLABLE(ModelLogReg);