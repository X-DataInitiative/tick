// License: BSD 3 clause

%{
#include "tick/linear_model/model_linreg.h"
%}

%include "model_generalized_linear.i";
%include "model_lipschitz.i";

template <class T>
class TModelLinReg : public virtual TModelGeneralizedLinear<T>, public TModelLipschitz<T> {
 public:
  TModelLinReg(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};

%rename(ModelLinReg) TModelLinReg<double>;
class ModelLinReg : public virtual TModelGeneralizedLinear<double>, public TModelLipschitz<double> {
 public:
  ModelLinReg(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelLinReg<double> ModelLinReg;

%rename(ModelLinRegDouble) TModelLinReg<double>;
class TModelLinReg<double> : public virtual TModelGeneralizedLinear<double>, public TModelLipschitz<double> {
 public:
  TModelLinReg(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelLinReg<double> ModelLinRegDouble;

%rename(ModelLinRegFloat) TModelLinReg<float>;
class TModelLinReg<float> : public virtual TModelGeneralizedLinear<float>, public TModelLipschitz<float> {
 public:
  TModelLinReg(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelLinReg<float> ModelLinRegFloat;
