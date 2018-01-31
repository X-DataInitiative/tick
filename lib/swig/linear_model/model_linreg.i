// License: BSD 3 clause

%{
#include "tick/linear_model/model_linreg.h"
%}

%include "model_generalized_linear.i";
%include "model_lipschitz.i";

template <class T, class K>
class TModelLinReg : public virtual TModelGeneralizedLinear<T, K>, public TModelLipschitz<T, K> {
 public:
  TModelLinReg(
    const std::shared_ptr<BaseArray2d<K> > features,
    const std::shared_ptr<SArray<K> > labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};

%rename(ModelLinRegDouble) TModelLinReg<double, double>;
class TModelLinReg<double, double> : public virtual TModelGeneralizedLinear<double, double>, public TModelLipschitz<double, double> {
 public:
  TModelLinReg(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelLinReg<double, double> ModelLinRegDouble;

%rename(ModelLinRegFloat) TModelLinReg<float, float>;
class TModelLinReg<float, float> : public virtual TModelGeneralizedLinear<float, float>, public TModelLipschitz<float, float> {
 public:
  TModelLinReg(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelLinReg<float, float> ModelLinRegFloat;

class ModelLinReg : public ModelLinRegDouble {
 public:
  ModelLinReg(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};

