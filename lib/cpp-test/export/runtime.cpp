// License: BSD 3 clause

/*********************************
 * Author: Philip Deegan
 * Date: 19/02/2017
 *
 * This file exists to check that
 *  all symbols are exported
 *  so at link time errors are seen
 *  and python modules load properly
 *********************************/

#include "tick/solver/adagrad.h"
#include "tick/solver/saga.h"
#include "tick/solver/sdca.h"
#include "tick/solver/sgd.h"
#include "tick/solver/svrg.h"

#include "tick/linear_model/model_linreg.h"
#include "tick/linear_model/model_logreg.h"

#include "tick/prox/prox_binarsity.h"
#include "tick/prox/prox_elasticnet.h"
#include "tick/prox/prox_equality.h"
#include "tick/prox/prox_l1.h"
#include "tick/prox/prox_l1w.h"
#include "tick/prox/prox_l2.h"
#include "tick/prox/prox_l2sq.h"
#include "tick/prox/prox_multi.h"
#include "tick/prox/prox_positive.h"
#include "tick/prox/prox_separable.h"
#include "tick/prox/prox_slope.h"
#include "tick/prox/prox_sorted_l1.h"
#include "tick/prox/prox_tv.h"
#include "tick/prox/prox_with_groups.h"
#include "tick/prox/prox_zero.h"

#include "tick/robust/model_absolute_regression.h"

#include "tick/survival/model_coxreg_partial_lik.h"
#include "tick/survival/model_sccs.h"

#include "tick/hawkes/model/model_hawkes_expkern_leastsq_single.h"
#include "tick/hawkes/model/model_hawkes_sumexpkern_loglik_single.h"

#include "tick/hawkes/simulation/simu_hawkes.h"
#include "tick/hawkes/simulation/simu_poisson_process.h"

#ifdef _MKN_WITH_MKN_KUL_
#include "kul/log.hpp"
#include "kul/signal.hpp"
#endif

#define DEBUG std::cout << __LINE__ << std::endl

void run(const std::function<void()>& func) {
  try {
    func();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  } catch (...) {
    DEBUG;
  }
}

SArrayDoublePtr get_labels() {
  ArrayDouble labels{-1.76, 2.6, -0.7, -1.84, -1.88, -1.78, 2.52};
  return labels.as_sarray_ptr();
}

SArrayDouble2dPtr get_features() {
  ulong n_samples = 7;
  ulong n_features = 5;

  ArrayDouble features_data{0.55,  2.04,  0.78,  -0.00, 0.00,  -0.00, -2.62,
                            -0.00, 0.00,  0.31,  -0.64, 0.94,  0.00,  0.55,
                            -0.14, 0.93,  0.00,  0.00,  -0.00, -2.39, 1.13,
                            0.05,  -1.50, -0.50, -1.41, 1.41,  1.10,  -0.00,
                            0.12,  0.00,  -0.00, -1.33, -0.00, 0.85,  3.03};

  ArrayDouble2d features(n_samples, n_features);
  for (int i = 0; i < features_data.size(); ++i) {
    features[i] = features_data[i];
  }
  return features.as_sarray2d_ptr();
}

#ifdef ADD_MAIN
int main(int argc, char** argv) {
#ifdef _MKN_WITH_MKN_KUL_
  kul::Signal sig;
#endif
  SArrayULongPtr start, length;
  std::shared_ptr<SArray<double>> weights;

  SArrayDouble2dPtr features_ptr = get_features();
  SArrayDoublePtr labels_ptr = get_labels();

  ulong n_samples = features_ptr->n_rows();
  ulong n_features = features_ptr->n_cols();
  (void) n_samples;
  (void) n_features;

  try {
    auto a_linreg =
      std::make_shared<TModelLinReg<double, std::atomic<double>>>(features_ptr,
                                                         labels_ptr, false, 1);

    auto linreg = std::make_shared<TModelLinReg<double>>(features_ptr,
                                                         labels_ptr, false, 1);
    auto logreg = std::make_shared<TModelLogReg<double>>(features_ptr,
                                                         labels_ptr, false, 1);

    run([&]() { TProxZero<double> t(1); });
    run([&]() { TProxPositive<double> t(1); });
    run([&]() { TProxEquality<double> t(1, true); });
    run([&]() { TProxElasticNet<double> t(1, 1, true); });
    run([&]() { TProxL1<double> t(1, true); });
    run([&]() { TProxL1w<double> t(1, weights, true); });
    run([&]() { TProxL2<double> t(1, true); });
    run([&]() { TProxL2Sq<double> t(1, true); });
    run([&]() { TProxSeparable<double> t(1, true); });
    run([&]() { TProxSeparable<double, std::atomic<double>> t(1, true); });
    run([&]() { TProxSlope<double> t(1, 1, true); });
    run([&]() { TProxSortedL1<double> t(1, {}, true); });
    run([&]() { TProxTV<double> t(1, true); });
    run([&]() { TProxWithGroups<double> t(1, start, length, true); });
    run([&]() { TProxBinarsity<double> t(1, start, length, true); });
    run([&]() { TProxMulti<double> t; });
    run([&]() {
      TAdaGrad<double> svrg(n_samples, 0, RandType::unif,
                            linreg->get_lip_max() / 100, 1309);
    });
    run([&]() {
      TSAGA<double> svrg(n_samples, 0, RandType::unif,
                         linreg->get_lip_max() / 100, 1309);
    });
    run([&]() { TSDCA<double> svrg(1, n_samples, 0, RandType::unif, 1309); });
    run([&]() {
      TSGD<double> svrg(n_samples, 0, RandType::unif,
                        linreg->get_lip_max() / 100, 1309);
    });
    run([&]() {
      TSVRG<double> svrg(n_samples, 0, RandType::unif,
                         linreg->get_lip_max() / 100, 1309);
    });

    run([&]() {
      SBaseArrayDouble2dPtrList1D features;
      SArrayIntPtrList1D labels;
      SBaseArrayULongPtr censoring;
      SArrayULongPtr n_lags;
      ModelSCCS t(features, labels, censoring, n_lags);
    });
    run([&]() {
      std::shared_ptr<BaseArray2d<double>> features;
      std::shared_ptr<SArray<double>> times;
      SArrayUShortPtr censoring;
      TModelCoxRegPartialLik<double> t(features, times, censoring);
    });

    run([&]() {
      TModelAbsoluteRegression<double> t(features_ptr, labels_ptr, false, 1);
      std::cout << "TModelAbsoluteRegression<double>.get_class_name() "
                << t.get_class_name() << std::endl;
    });

    run([&]() {
      ModelHawkesExpKernLeastSqSingle t(features_ptr, 1, 1);
      std::cout << "ModelHawkesExpKernLeastSqSingle<double>.get_class_name() "
                << t.get_class_name() << std::endl;
    });

    run([&]() { Hawkes t(1, 1); });
    run([&]() { Poisson t(1, 1); });

  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  } catch (...) {
    DEBUG;
  }
  return 0;
}
#endif  // ADD_MAIN
