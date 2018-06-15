// License: BSD 3 clause

%{
#include "tick/linear_model/model_hinge.h"
%}

%include "model_generalized_linear.i";

template <class T, class K = T>
class DLL_PUBLIC TModelHinge : public virtual TModelGeneralizedLinear<T, K> {
 public:
  TModelHinge();
  TModelHinge(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads
  );

  bool compare(const TModelHinge<T, K> &that);
};

%rename(ModelHingeDouble) TModelHinge<double, double>;
class ModelHingeDouble : public virtual TModelGeneralizedLinear<double, double>{
 public:
  ModelHingeDouble();
  ModelHingeDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );

  bool compare(const ModelHingeDouble &that);
};
typedef TModelHinge<double, double> ModelHingeDouble;
TICK_MAKE_PICKLABLE(ModelHingeDouble);

%rename(ModelHingeFloat) TModelHinge<float, float>;
class ModelHingeFloat : public virtual TModelGeneralizedLinear<float, float>{
 public:
  ModelHingeFloat();
  ModelHingeFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );

  bool compare(const ModelHingeFloat &that);
};
typedef TModelHinge<float, float> ModelHingeFloat;
TICK_MAKE_PICKLABLE(ModelHingeFloat);
