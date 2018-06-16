// License: BSD 3 clause

%{
#include "tick/linear_model/model_logreg.h"
%}

%include "model_generalized_linear.i";
%include "model_lipschitz.i";

template <class T, class K = T>
class TModelLogReg : public virtual TModelGeneralizedLinear<T, K>, public TModelLipschitz<T, K> {
 public:
  TModelLogReg();
  TModelLogReg(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads = 1
  );

  bool compare(const TModelLogReg<T, K> &that);

  static void sigmoid(const Array<T> &x, Array<T> &out);
  static void logistic(const Array<T> &x, Array<T> &out);
};

%rename(ModelLogRegDouble) TModelLogReg<double, double>;
class ModelLogRegDouble : public virtual TModelGeneralizedLinear<double, double>, public TModelLipschitz<double, double> {
 public:
  ModelLogRegDouble();
  ModelLogRegDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads
  );

  bool compare(const ModelLogRegDouble &that);

  static void sigmoid(const ArrayDouble &x, ArrayDouble &out);
  static void logistic(const ArrayDouble &x, ArrayDouble &out);
};
typedef TModelLogReg<double, double> ModelLogRegDouble;
TICK_MAKE_PICKLABLE(ModelLogRegDouble);

%rename(ModelLogRegFloat) TModelLogReg<float, float>;
class ModelLogRegFloat : public virtual TModelGeneralizedLinear<float, float>, public TModelLipschitz<float, float> {
 public:
  ModelLogRegFloat();
  ModelLogRegFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );

  bool compare(const ModelLogRegFloat &that);

  static void sigmoid(const Array<float> &x, Array<float> &out);
  static void logistic(const Array<float> &x, Array<float> &out);
};
typedef TModelLogReg<float, float> ModelLogRegFloat;
TICK_MAKE_PICKLABLE(ModelLogRegFloat);



%rename(ModelLogRegAtomicDouble) TModelLogReg<double, std::atomic<double>>;
class ModelLogRegAtomicDouble : public virtual TModelGeneralizedLinear<double, std::atomic<double>>, public TModelLipschitz<double, std::atomic<double>> {
 public:
  ModelLogRegAtomicDouble();
  ModelLogRegAtomicDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads
  );

  bool compare(const TModelLogReg<double, std::atomic<double>> &that);

  static void sigmoid(const ArrayDouble &x, ArrayDouble &out);
  static void logistic(const ArrayDouble &x, ArrayDouble &out);
};
typedef TModelLogReg<double, std::atomic<double>> ModelLogRegAtomicDouble;


%rename(ModelLogRegAtomicFloat) TModelLogReg<float, std::atomic<float>>;
class TModelLogReg<float, std::atomic<float>> : public virtual TModelGeneralizedLinear<float, std::atomic<float>>, public TModelLipschitz<float, std::atomic<float>> {
 public:
  ModelLogRegAtomicFloat();
  ModelLogRegAtomicFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const int n_threads
  );

  bool compare(const TModelLogReg<float, std::atomic<float>> &that);

  static void sigmoid(const Array<float> &x, Array<float> &out);
  static void logistic(const Array<float> &x, Array<float> &out);
};
typedef TModelLogReg<float, std::atomic<float>> ModelLogRegAtomicFloat;
