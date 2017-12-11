// License: BSD 3 clause


%{
#include "tick/base_model/model_labels_features.h"
%}


class ModelLabelsFeatures : public virtual Model {

 public:
  ModelLabelsFeatures(const SBaseArrayDouble2dPtr features,
                      const SArrayDoublePtr labels);

  virtual unsigned long get_n_samples() const;
  virtual unsigned long get_n_features() const;
};
