// License: BSD 3 clause

%{
#include "tick/linear_model/model_linreg.h"
%}

%include "tick/base_model/model_lipschitz.i";
%include "tick/base_model/model_generalized_linear.i";

template <class T, class K = T>
class TModelLinReg : public virtual TModelGeneralizedLinear<T, K>, public TModelLipschitz<T, K> {
 public:
  TModelLinReg();
  TModelLinReg(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads = 1
  );

  bool compare(const TModelLinReg<T, K> &that);
};

%rename(ModelLinRegDouble) TModelLinReg<double, double>;
class ModelLinRegDouble : public virtual TModelGeneralizedLinear<double, double>, public TModelLipschitz<double, double> {
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
typedef TModelLinReg<double, double> ModelLinRegDouble;
TICK_MAKE_PICKLABLE(ModelLinRegDouble);

%rename(ModelLinRegFloat) TModelLinReg<float, float>;
class ModelLinRegFloat : public virtual TModelGeneralizedLinear<float, float>, public TModelLipschitz<float, float> {
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
typedef TModelLinReg<float, float> ModelLinRegFloat;
TICK_MAKE_PICKLABLE(ModelLinRegFloat);
