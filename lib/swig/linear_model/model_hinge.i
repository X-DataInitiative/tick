// License: BSD 3 clause

%{
#include "tick/linear_model/model_hinge.h"
%}

%include "model_generalized_linear.i";

template <class T>
class DLL_PUBLIC TModelHinge : public virtual TModelGeneralizedLinear<T> {
 public:
  TModelHinge();
  TModelHinge(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads
  );

  bool compare(const TModelHinge<T> &that);
};

%rename(ModelHingeDouble) TModelHinge<double>;
class ModelHingeDouble : public virtual TModelGeneralizedLinear<double>{
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
typedef TModelHinge<double> ModelHingeDouble;
TICK_MAKE_PICKLABLE(ModelHingeDouble);

%rename(ModelHingeFloat) TModelHinge<float>;
class ModelHingeFloat : public virtual TModelGeneralizedLinear<float>{
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
typedef TModelHinge<float> ModelHingeFloat;
TICK_MAKE_PICKLABLE(ModelHingeFloat);
