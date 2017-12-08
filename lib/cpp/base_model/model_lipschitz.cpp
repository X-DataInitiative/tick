// License: BSD 3 clause

//
// Created by Stéphane GAIFFAS on 17/03/2016.
//

#include "tick/base_model/model_lipschitz.h"

ModelLipschitz::ModelLipschitz() : Model() {
  ready_lip_consts = false;
  ready_lip_max = false;
  ready_lip_mean = false;
  lip_mean = 0;
  lip_max = 0;
}

double ModelLipschitz::get_lip_max() {
  if (ready_lip_max) {
    return lip_max;
  } else {
    compute_lip_consts();
    lip_max = lip_consts.max();
    ready_lip_max = true;
    return lip_max;
  }
}

double ModelLipschitz::get_lip_mean() {
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
