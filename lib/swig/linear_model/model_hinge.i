// License: BSD 3 clause

%{
#include "tick/linear_model/model_hinge.h"
%}

%include "model_generalized_linear.i";

template <class T>
class DLL_PUBLIC TModelHinge : public virtual TModelGeneralizedLinear<T> {
 public:
  TModelHinge(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads
  );
};

%rename(ModelHinge) TModelHinge<double>;
class ModelHinge : public virtual TModelGeneralizedLinear<double>{
 public:
  ModelHinge(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelHinge<double> ModelHinge;

%rename(ModelHingeDouble) TModelHinge<double>;
class ModelHingeDouble : public virtual TModelGeneralizedLinear<double>{
 public:
  ModelHingeDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelHinge<double> ModelHingeDouble;

%rename(ModelHingeFloat) TModelHinge<float>;
class ModelHingeFloat : public virtual TModelGeneralizedLinear<float>{
 public:
  ModelHingeFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelHinge<float> ModelHingeFloat;
