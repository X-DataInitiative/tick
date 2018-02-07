// License: BSD 3 clause

%{
#include "tick/linear_model/model_smoothed_hinge.h"
%}

%include "model_generalized_linear.i";
%include "model_lipschitz.i";

template <class T>
class DLL_PUBLIC TModelSmoothedHinge : public virtual TModelGeneralizedLinear<T>, public TModelLipschitz<T> {
 public:
  TModelSmoothedHinge(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const T smoothness = 1,
    const int n_threads = 1);

  T get_smoothness() const;

  void set_smoothness(T smoothness);
};

%rename(ModelSmoothedHinge) TModelSmoothedHinge<double>;
class ModelSmoothedHinge : public virtual TModelGeneralizedLinear<double>,
                    public TModelLipschitz<double> {
 public:
  ModelSmoothedHinge(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const double smoothness = 1,
    const int n_threads = 1
  );
  double get_smoothness() const;

  void set_smoothness(double smoothness);
};
typedef TModelSmoothedHinge<double> ModelSmoothedHinge;

%rename(ModelSmoothedHingeDouble) TModelSmoothedHinge<double>;
class ModelSmoothedHingeDouble : public virtual TModelGeneralizedLinear<double>,
                    public TModelLipschitz<double> {
 public:
  ModelSmoothedHingeDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const double smoothness = 1,
    const int n_threads = 1
  );
  double get_smoothness() const;

  void set_smoothness(double smoothness);
};
typedef TModelSmoothedHinge<double> ModelSmoothedHingeDouble;

%rename(ModelSmoothedHingeFloat) TModelSmoothedHinge<float>;
class ModelSmoothedHingeFloat : public virtual TModelGeneralizedLinear<float>,
                    public TModelLipschitz<float> {
 public:
  ModelSmoothedHingeFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const float smoothness = 1,
    const int n_threads = 1
  );
  float get_smoothness() const;

  void set_smoothness(float smoothness);
};
typedef TModelSmoothedHinge<double> ModelSmoothedHingeFloat;
