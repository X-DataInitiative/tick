// License: BSD 3 clause

//
// Created by Stéphane GAIFFAS on 17/03/2016.
//

#include "tick/base_model/model_lipschitz.h"

template <class T, class K>
TModelLipschitz<T, K>::TModelLipschitz() : TModel<T, K>() {
  ready_lip_consts = false;
  ready_lip_max = false;
  ready_lip_mean = false;
  lip_mean = 0;
  lip_max = 0;
}

template <class T, class K>
T TModelLipschitz<T, K>::get_lip_max() {
  if (ready_lip_max) {
    return lip_max;
  } else {
    compute_lip_consts();
    lip_max = lip_consts.max();
    ready_lip_max = true;
    return lip_max;
  }
}

template <class T, class K>
T TModelLipschitz<T, K>::get_lip_mean() {
  if (ready_lip_mean) {
    return lip_mean;
  } else {
    compute_lip_consts();
    // TODO: no mean method in array.h, really ?!?
    lip_mean = lip_consts.sum() / lip_consts.size();
    ready_lip_mean = true;
    return lip_mean;
  }
}

template class TModelLipschitz<double, double>;
template class TModelLipschitz<float, float>;

template class TModelLipschitz<double, std::atomic<double>>;
template class TModelLipschitz<float, std::atomic<float>>;
