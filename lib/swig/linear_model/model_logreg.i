// License: BSD 3 clause

%{
#include "tick/linear_model/model_logreg.h"
%}

%include "model_generalized_linear.i";
%include "model_lipschitz.i";

template <class T, class K>
class TModelLogReg : public virtual TModelGeneralizedLinear<T, K>, public TModelLipschitz<T, K> {
 public:
  TModelLogReg(
    const std::shared_ptr<BaseArray2d<K> > features,
    const std::shared_ptr<SArray<K> > labels,
    const bool fit_intercept,
    const int n_threads = 1
  );

  static void sigmoid(const Array<K> &x, Array<K> &out);
  static void logistic(const Array<K> &x, Array<K> &out);
};

%rename(ModelLogRegDouble) TModelLogReg<double, double>;
class TModelLogReg<double, double> : public virtual TModelGeneralizedLinear<double, double>, public TModelLipschitz<double, double> {
 public:
  TModelLogReg(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads
  );
  static void sigmoid(const ArrayDouble &x, ArrayDouble &out);
  static void logistic(const ArrayDouble &x, ArrayDouble &out);
};
typedef TModelLogReg<double, double> ModelLogRegDouble;

%rename(ModelLogRegFloat) TModelLogReg<float, float>;
class TModelLogReg<float, float> : public virtual TModelGeneralizedLinear<float, float>, public TModelLipschitz<float, float> {
 public:
  TModelLogReg(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
  static void sigmoid(const Array<float> &x, Array<float> &out);
  static void logistic(const Array<float> &x, Array<float> &out);
};
typedef TModelLogReg<float, float> ModelLogRegFloat;

class ModelLogReg : public ModelLogRegDouble {
 public:
  ModelLogReg(const SBaseArrayDouble2dPtr features,
              const SArrayDoublePtr labels,
              const bool fit_intercept,
              const int n_threads);

  static void sigmoid(const ArrayDouble &x, ArrayDouble &out);
  static void logistic(const ArrayDouble &x, ArrayDouble &out);
};