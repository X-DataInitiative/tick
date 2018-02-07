// License: BSD 3 clause

%{
#include "tick/linear_model/model_quadratic_hinge.h"
%}

%include "model_generalized_linear.i";
%include "model_lipschitz.i";

template <class T>
class TModelQuadraticHinge : public virtual TModelGeneralizedLinear<T>,
                    public TModelLipschitz<T> {
 public:
  TModelQuadraticHinge(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads
  );
};

%rename(ModelQuadraticHinge) TModelQuadraticHinge<double>;
class ModelQuadraticHinge : public virtual TModelGeneralizedLinear<double>,
                    public TModelLipschitz<double> {
 public:
  ModelQuadraticHinge(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelQuadraticHinge<double> ModelQuadraticHinge;

%rename(ModelQuadraticHingeDouble) TModelQuadraticHinge<double>;
class ModelQuadraticHingeDouble : public virtual TModelGeneralizedLinear<double>,
                    public TModelLipschitz<double> {
 public:
  ModelQuadraticHingeDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelQuadraticHinge<double> ModelQuadraticHingeDouble;

%rename(ModelQuadraticHingeFloat) TModelQuadraticHinge<float>;
class ModelQuadraticHingeFloat : public virtual TModelGeneralizedLinear<float>,
                    public TModelLipschitz<float> {
 public:
  ModelQuadraticHingeFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelQuadraticHinge<float> ModelQuadraticHingeFloat;
