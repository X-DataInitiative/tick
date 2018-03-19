// License: BSD 3 clause

%{
#include "tick/robust/model_modified_huber.h"
%}

template <class T>
class TModelModifiedHuber : public virtual TModelGeneralizedLinear<T>,
                    public TModelLipschitz<T> {
 public:
  TModelModifiedHuber();
  TModelModifiedHuber(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels,
    const bool fit_intercept,
    const int n_threads
  );
  bool compare(const TModelModifiedHuber<T> &that);
};

%rename(ModelModifiedHuberDouble) TModelModifiedHuber<double>;
class ModelModifiedHuberDouble : public virtual TModelGeneralizedLinear<double>,
                    public TModelLipschitz<double> {
 public:
  ModelModifiedHuberDouble();
  ModelModifiedHuberDouble(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
  bool compare(const ModelModifiedHuberDouble &that);
};
typedef TModelModifiedHuber<double> ModelModifiedHuberDouble;
TICK_MAKE_PICKLABLE(ModelModifiedHuberDouble);

%rename(ModelModifiedHuberFloat) TModelModifiedHuber<float>;
class ModelModifiedHuberFloat : public virtual TModelGeneralizedLinear<float>,
                    public TModelLipschitz<float> {
 public:
  ModelModifiedHuberFloat();
  ModelModifiedHuberFloat(
    const SBaseArrayFloat2dPtr features,
    const SArrayFloatPtr labels,
    const bool fit_intercept,
    const int n_threads = 1
  );
  bool compare(const ModelModifiedHuberFloat &that);
};
typedef TModelModifiedHuber<float> ModelModifiedHuberFloat;
TICK_MAKE_PICKLABLE(ModelModifiedHuberFloat);
