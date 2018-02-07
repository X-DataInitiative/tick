// License: BSD 3 clause

%{
#include "tick/linear_model/model_logreg.h"
%}

%include "model_generalized_linear.i";
%include "model_lipschitz.i";

template <class T>
class TModelLogReg : public virtual TModelGeneralizedLinear<T>, public TModelLipschitz<T> {
 public:
  TModelLogReg(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads = 1
  );

  static void sigmoid(const Array<T> &x, Array<T> &out);
  static void logistic(const Array<T> &x, Array<T> &out);
};

%rename(ModelLogReg) TModelLogReg<double>;
class ModelLogReg : public virtual TModelGeneralizedLinear<double>, public TModelLipschitz<double> {
 public:
  ModelLogReg(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads
  );
  static void sigmoid(const ArrayDouble &x, ArrayDouble &out);
  static void logistic(const ArrayDouble &x, ArrayDouble &out);
};
typedef TModelLogReg<double> ModelLogReg;

%rename(ModelLogRegDouble) TModelLogReg<double>;
class TModelLogReg<double> : public virtual TModelGeneralizedLinear<double>, public TModelLipschitz<double> {
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
typedef TModelLogReg<double> ModelLogRegDouble;

%rename(ModelLogRegFloat) TModelLogReg<float>;
class TModelLogReg<float> : public virtual TModelGeneralizedLinear<float>, public TModelLipschitz<float> {
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
typedef TModelLogReg<float> ModelLogRegFloat;
