// License: BSD 3 clause


%{
#include "tick/base_model/model_labels_features.h"
%}

%include "model.i"

template <class T>
class TModelLabelsFeatures : public virtual TModel<T> {
 public:
  TModelLabelsFeatures(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels
  );
  virtual unsigned long get_n_samples() const;
  virtual unsigned long get_n_features() const;
};

%template(ModelLabelsFeatures) TModelLabelsFeatures<double>;
typedef TModelLabelsFeatures<double> ModelLabelsFeatures;

%template(ModelLabelsFeaturesDouble) TModelLabelsFeatures<double>;
typedef TModelLabelsFeatures<double> ModelLabelsFeaturesDouble;

%template(ModelLabelsFeaturesFloat) TModelLabelsFeatures<float>;
typedef TModelLabelsFeatures<float> ModelLabelsFeaturesFloat;

