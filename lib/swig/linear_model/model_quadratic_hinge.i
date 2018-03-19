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
  TModelQuadraticHinge();
  TModelQuadraticHinge(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads
  );

  bool compare(const TModelQuadraticHinge<T> &that);
};

%rename(ModelQuadraticHingeDouble) TModelQuadraticHinge<double>;
class ModelQuadraticHingeDouble : public virtual TModelGeneralizedLinear<double>,
                    public TModelLipschitz<double> {
 public:
  ModelQuadraticHingeDouble();
  ModelQuadraticHingeDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );

  bool compare(const ModelQuadraticHingeDouble &that);
};
typedef TModelQuadraticHinge<double> ModelQuadraticHingeDouble;
TICK_MAKE_PICKLABLE(ModelQuadraticHingeDouble);

%rename(ModelQuadraticHingeFloat) TModelQuadraticHinge<float>;
class ModelQuadraticHingeFloat : public virtual TModelGeneralizedLinear<float>,
                    public TModelLipschitz<float> {
 public:
  ModelQuadraticHingeFloat();
  ModelQuadraticHingeFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );

  bool compare(const ModelQuadraticHingeFloat &that);
};
typedef TModelQuadraticHinge<float> ModelQuadraticHingeFloat;
TICK_MAKE_PICKLABLE(ModelQuadraticHingeFloat);
