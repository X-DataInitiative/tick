// License: BSD 3 clause

%{
#include "tick/survival/model_coxreg_partial_lik.h"
%}

%include "model_lipschitz.i";

template <class T, class K = T>
class TModelCoxRegPartialLik : public TModel<T, K> {
 public:
  TModelCoxRegPartialLik(const std::shared_ptr<BaseArray2d<T> > features,
                         const std::shared_ptr<SArray<T> > times,
                         const SArrayUShortPtr censoring);
};

%rename(ModelCoxRegPartialLik) TModelCoxRegPartialLik<double>;
class TModelCoxRegPartialLik<double> : public TModel<double> {
 public:
  ModelCoxRegPartialLik(const SBaseArrayDouble2dPtr features,
                              const SArrayDoublePtr times,
                              const SArrayUShortPtr censoring);
};
typedef TModelCoxRegPartialLik<double> ModelCoxRegPartialLik;

%rename(ModelCoxRegPartialLikDouble) TModelCoxRegPartialLik<double>;
class TModelCoxRegPartialLik<double> : public TModel<double> {
 public:
  ModelCoxRegPartialLikDouble(const SBaseArrayDouble2dPtr features,
                              const SArrayDoublePtr times,
                              const SArrayUShortPtr censoring);
};
typedef TModelCoxRegPartialLik<double> ModelCoxRegPartialLikDouble;

%rename(ModelCoxRegPartialLikFloat) TModelCoxRegPartialLik<float>;
class TModelCoxRegPartialLik<float> : public TModel<float> {
 public:
  ModelCoxRegPartialLikFloat(const SBaseArrayFloat2dPtr features,
                             const SArrayFloatPtr times,
                             const SArrayUShortPtr censoring);
};
typedef TModelCoxRegPartialLik<float> ModelCoxRegPartialLikFloat;
