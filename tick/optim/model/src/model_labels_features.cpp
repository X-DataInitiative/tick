//
// Created by Stéphane GAIFFAS on 06/12/2015.
//

#include "model_labels_features.h"

ModelLabelsFeatures::ModelLabelsFeatures(SBaseArrayDouble2dPtr features,
                                         SArrayDoublePtr labels)
    : n_samples(labels.get() ? labels->size() : 0),
      n_features(features.get() ? features->n_cols() : 0),
      labels(labels),
      features(features) {
  if (labels.get() && labels->size() != features->n_rows()) {
    std::stringstream ss;
    ss << "In ModelLabelsFeatures, number of labels is " << labels->size();
    ss << " while the features matrix has " << features->n_rows() << " rows.";
    throw std::invalid_argument(ss.str());
  }
}
