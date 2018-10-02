// License: BSD 3 clause

%{
#include "tick/robust/model_huber.h"
%}

template <class T>
class TModelHuber : public virtual TModelGeneralizedLinear<T, K>, public TModelLipschitz<T, K> {
 public:
  TModelHuber();
  TModelHuber(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
  virtual T get_threshold(void) const;
  virtual void set_threshold(const T threshold);
  bool compare(const TModelHuber<T, K> &that);
};

%rename(ModelHuberDouble) TModelHuber<double, double>;
class ModelHuberDouble : public virtual TModelGeneralizedLinear<double, double>, public TModelLipschitz<double, double> {
 public:
  ModelHuberDouble();
  ModelHuberDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const double threshold,
    const int n_threads = 1
  );
  virtual double get_threshold(void) const;
  virtual void set_threshold(const double threshold);
  bool compare(const ModelHuberDouble &that);
};
typedef TModelHuber<double, double> ModelHuberDouble;
TICK_MAKE_PICKLABLE(ModelHuberDouble);

%rename(ModelHuberFloat) TModelHuber<float, float>;
class ModelHuberFloat : public virtual TModelGeneralizedLinear<float, float>, public TModelLipschitz<float, float> {
 public:
  ModelHuberFloat();
  ModelHuberFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const float threshold,
    const int n_threads = 1
  );
  virtual float get_threshold(void) const;
  virtual void set_threshold(const float threshold);
  bool compare(const ModelHuberFloat &that);
};
typedef TModelHuber<float, float> ModelHuberFloat;
TICK_MAKE_PICKLABLE(ModelHuberFloat);
