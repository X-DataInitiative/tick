// License: BSD 3 clause

%{
#include "tick/survival/model_coxreg_partial_lik.h"
%}

%include "tick/base_model/model_lipschitz.i";

template <class T, class K = T>
class TModelCoxRegPartialLik : public TModel<T, K> {
 public:
  TModelCoxRegPartialLik(const std::shared_ptr<SArray2d<T> > features,
                         const std::shared_ptr<SArray<T> > times,
                         const SArrayUShortPtr censoring);

  bool compare(const TModelCoxRegPartialLik &that);
};

%rename(ModelCoxRegPartialLikDouble) TModelCoxRegPartialLik<double>;
class ModelCoxRegPartialLikDouble : public TModel<double> {
 public:
  ModelCoxRegPartialLikDouble();
  ModelCoxRegPartialLikDouble(const SBaseArrayDouble2dPtr features,
                              const SArrayDoublePtr times,
                              const SArrayUShortPtr censoring);

  bool compare(const ModelCoxRegPartialLikDouble &that);
};
typedef TModelCoxRegPartialLik<double> ModelCoxRegPartialLikDouble;
TICK_MAKE_PICKLABLE(ModelCoxRegPartialLikDouble);

%rename(ModelCoxRegPartialLikFloat) TModelCoxRegPartialLik<float>;
class ModelCoxRegPartialLikFloat : public TModel<float> {
 public:
  ModelCoxRegPartialLikFloat();
  ModelCoxRegPartialLikFloat(const SBaseArrayFloat2dPtr features,
                             const SArrayFloatPtr times,
                             const SArrayUShortPtr censoring);

  bool compare(const ModelCoxRegPartialLikFloat &that);
};
typedef TModelCoxRegPartialLik<float> ModelCoxRegPartialLikFloat;
TICK_MAKE_PICKLABLE(ModelCoxRegPartialLikFloat);
