// License: BSD 3 clause

%{
#include "tick/linear_model/model_linreg.h"
%}

%include "model_generalized_linear.i";
%include "model_lipschitz.i";

template <class T>
class TModelLinReg : public virtual TModelGeneralizedLinear<T>, public TModelLipschitz<T> {
 public:
  TModelLinReg();
  TModelLinReg(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads = 1
  );

  bool compare(const TModelLinReg<T> &that);
};

%rename(ModelLinRegDouble) TModelLinReg<double>;
class ModelLinRegDouble : public virtual TModelGeneralizedLinear<double>, public TModelLipschitz<double> {
 public:
  ModelLinRegDouble();
  ModelLinRegDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );

  bool compare(const ModelLinRegDouble &that);
};
typedef TModelLinReg<double> ModelLinRegDouble;
TICK_MAKE_PICKLABLE(ModelLinRegDouble);

%rename(ModelLinRegFloat) TModelLinReg<float>;
class ModelLinRegFloat : public virtual TModelGeneralizedLinear<float>, public TModelLipschitz<float> {
 public:
  ModelLinRegFloat();
  ModelLinRegFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );

  bool compare(const ModelLinRegFloat &that);
};
typedef TModelLinReg<float> ModelLinRegFloat;
TICK_MAKE_PICKLABLE(ModelLinRegFloat);
