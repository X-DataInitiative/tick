// License: BSD 3 clause

%{
#include "tick/robust/model_modified_huber.h"
%}

template <class T>
class TModelModifiedHuber : public virtual TModelGeneralizedLinear<T>,
                    public TModelLipschitz<T> {
 public:
  TModelModifiedHuber(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads
  );
};

%rename(ModelModifiedHuber) TModelModifiedHuber<double>;
class ModelModifiedHuber : public virtual TModelGeneralizedLinear<double>,
                    public TModelLipschitz<double> {
 public:
  ModelModifiedHuber(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelModifiedHuber<double> ModelModifiedHuber;

%rename(ModelModifiedHuberDouble) TModelModifiedHuber<double>;
class ModelModifiedHuberDouble : public virtual TModelGeneralizedLinear<double>,
                    public TModelLipschitz<double> {
 public:
  ModelModifiedHuberDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelModifiedHuber<double> ModelModifiedHuberDouble;

%rename(ModelModifiedHuberFloat) TModelModifiedHuber<float>;
class ModelModifiedHuberFloat : public virtual TModelGeneralizedLinear<float>,
                    public TModelLipschitz<float> {
 public:
  ModelModifiedHuberFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
};
typedef TModelModifiedHuber<float> ModelModifiedHuberFloat;
