// License: BSD 3 clause

%{
#include "tick/robust/model_epsilon_insensitive.h"
%}

template <class T>
class TModelEpsilonInsensitive : public virtual TModelGeneralizedLinear<T>{
 public:
  TModelEpsilonInsensitive();
  TModelEpsilonInsensitive(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
  virtual T get_threshold(void) const;
  virtual void set_threshold(const T threshold);

  bool compare(const TModelEpsilonInsensitive<T> &that);
};

%rename(ModelEpsilonInsensitiveDouble) TModelEpsilonInsensitive<double>;
class ModelEpsilonInsensitiveDouble : public virtual TModelGeneralizedLinear<double>{
 public:
  ModelEpsilonInsensitiveDouble();
  ModelEpsilonInsensitiveDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const double threshold,
    const int n_threads = 1
  );
  virtual double get_threshold(void) const;
  virtual void set_threshold(const double threshold);

  bool compare(const ModelEpsilonInsensitiveDouble &that);
};
typedef TModelEpsilonInsensitive<double> ModelEpsilonInsensitiveDouble;
TICK_MAKE_PICKLABLE(ModelEpsilonInsensitiveDouble);

%rename(ModelEpsilonInsensitiveFloat) TModelEpsilonInsensitive<float>;
class ModelEpsilonInsensitiveFloat : public virtual TModelGeneralizedLinear<float>{
 public:
  ModelEpsilonInsensitiveFloat();
  ModelEpsilonInsensitiveFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const float threshold,
    const int n_threads = 1
  );
  virtual float get_threshold(void) const;
  virtual void set_threshold(const float threshold);

  bool compare(const ModelEpsilonInsensitiveFloat &that);
};
typedef TModelEpsilonInsensitive<float> ModelEpsilonInsensitiveFloat;
TICK_MAKE_PICKLABLE(ModelEpsilonInsensitiveFloat);

