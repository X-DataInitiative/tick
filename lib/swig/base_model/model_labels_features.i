// License: BSD 3 clause


%{
#include "tick/base_model/model_labels_features.h"
%}

%include "model.i"

template <class T, class K>
class TModelLabelsFeatures : public virtual TModel<T, K> {
 public:
  TModelLabelsFeatures(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels
  );
  virtual unsigned long get_n_samples() const;
  virtual unsigned long get_n_features() const;
};

%template(ModelLabelsFeaturesDouble) TModelLabelsFeatures<double, double>;
typedef TModelLabelsFeatures<double, double> ModelLabelsFeaturesDouble;

%template(ModelLabelsFeaturesFloat) TModelLabelsFeatures<float, float>;
typedef TModelLabelsFeatures<float, float> ModelLabelsFeaturesFloat;

