// License: BSD 3 clause

%{
#include "tick/linear_model/model_linreg.h"
%}

%include "model_generalized_linear.i";
%include "model_lipschitz.i";

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


%rename(ModelLinRegAtomicDouble) TModelLinReg<double, std::atomic<double>>;
class ModelLinRegAtomicDouble : public virtual TModelGeneralizedLinear<double, std::atomic<double>>, public TModelLipschitz<double, std::atomic<double>> {
 public:
  ModelLinRegAtomicDouble();
  ModelLinRegAtomicDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );

  bool compare(const TModelLinReg<double, std::atomic<double>> &that);
};
typedef ModelLinRegAtomicDouble<double, std::atomic<double>> ModelLogRegAtomicDouble;

%rename(ModelLinRegAtomicFloat) TModelLinReg<float, std::atomic<float>>;
class ModelLinRegAtomicFloat : public virtual TModelGeneralizedLinear<float, std::atomic<float>>, public TModelLipschitz<float, std::atomic<float>> {
 public:
  ModelLinRegAtomicFloat();
  ModelLinRegAtomicFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );

  bool compare(const TModelLinReg<float, std::atomic<float>> &that);
};
typedef ModelLinRegAtomicFloat<float, std::atomic<float>> ModelLogRegAtomicFloat;
