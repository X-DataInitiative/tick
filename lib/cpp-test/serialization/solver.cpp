
#include "include.h"

#include "tick/solver/adagrad.h"
#include "tick/solver/sdca.h"
#include "tick/solver/sgd.h"
#include "tick/solver/saga.h"
#include "tick/solver/svrg.h"


template <class T, class MODEL>
void test_linear_separable(std::function<std::shared_ptr<MODEL>(
                  const std::shared_ptr<BaseArray2d<T> > &features,
                  const std::shared_ptr<SArray<T> > &labels)>
                  get_model) {
  auto features = get_features<T>();
  auto labels = get_labels<T>();
  ulong n_samples = features->n_rows();
  const auto BETA = 1e-10;
  const auto STRENGTH = (1. / n_samples) + BETA;
  auto model = get_model(features, labels);
  auto proxl1 = std::make_shared<TProxL1<T, K> >(STRENGTH, true);
  auto proxl1w = std::make_shared<TProxL1w<T, K> >(
      STRENGTH, get_proxl1w_weights<T>(), true);
  auto proxl2 =
      std::make_shared<TProxL2<T, K> >(STRENGTH, 0, model->get_n_coeffs(), 0);
  auto proxl2sq = std::make_shared<TProxL2Sq<T, K> >(STRENGTH, true);
  auto proxelas = std::make_shared<TProxElasticNet<T, K> >(
      STRENGTH, BETA / STRENGTH, 0, model->get_n_coeffs(), 0);
  auto proxeq = std::make_shared<TProxEquality<T, K> >(
      STRENGTH, 0, model->get_n_coeffs(), 0);
  auto proxtv =
      std::make_shared<TProxTV<T, K> >(STRENGTH, 0, model->get_n_coeffs(), 0);
  auto proxpos = std::make_shared<TProxPositive<T, K> >(STRENGTH);
  auto prox0 = std::make_shared<TProxZero<T, K> >(STRENGTH);
  {
    auto solver = std::make_shared<TAdaGrad<T, K> >(n_samples, 0,
                                                    RandType::unif, 1e3, -1);
    SCOPED_TRACE("");
    WITH_PROX(proxl1, proxl1w, proxl2, proxl2sq, proxelas, proxeq, proxtv,
              proxpos, prox0);
  }
  {
    auto solver = std::make_shared<TSDCA<T, K> >(n_samples);
    SCOPED_TRACE("");
    WITH_PROX(proxl1, proxl1w, proxl2, proxl2sq, proxelas, proxeq, proxtv,
              proxpos, prox0);
  }
  {
    auto solver =
        std::make_shared<TSGD<T, K> >(n_samples, 0, RandType::unif, 1e3, -1);
    SCOPED_TRACE("");
    WITH_PROX(proxl1, proxl1w, proxl2, proxl2sq, proxelas, proxeq, proxtv,
              proxpos, prox0);
  }
  {
    auto solver =
        std::make_shared<TSVRG<T, K> >(n_samples, 0, RandType::unif, 1e3, -1);
    SCOPED_TRACE("");
    WITH_PROX(proxl1, proxl1w, proxl2, proxl2sq, proxelas, proxeq, proxtv,
              proxpos, prox0);
  }
  {
    auto solver =
        std::make_shared<TSVRG<T, K> >(n_samples, 0, RandType::unif, 1e3, -1);
    SCOPED_TRACE("");
    WITH_PROX(proxl1, proxl1w, proxl2, proxl2sq, proxelas, proxeq, proxtv,
              proxpos, prox0);
  }
}

//####################### LIN REG ############################################
TEST(Model, LinRegDoubleSerializationJSON) {
  test_linear_separable<double, TModelLinReg<double> >(
      [](const SBaseArrayDouble2dPtr &features, const SArrayDoublePtr &labels) {
        return std::make_shared<TModelLinReg<double> >(features, labels, false,
                                                       1);
      });
}
TEST(Model, LinRegFloatSerializationJSON) {
  test_linear_separable<float, TModelLinReg<float> >([](const SBaseArrayFloat2dPtr &features,
                                           const SArrayFloatPtr &labels) {
    return std::make_shared<TModelLinReg<float> >(features, labels, false, 1);
  });
}
//####################### LIN REG ############################################

//####################### LOG REG ############################################
TEST(Model, LogRegDoubleSerializationJSON) {
  test_linear_separable<double, TModelLogReg<double> >(
      [](const SBaseArrayDouble2dPtr &features, const SArrayDoublePtr &labels) {
        return std::make_shared<TModelLogReg<double> >(features, labels, false,
                                                       1);
      });
}

TEST(Model, LogRegFloatSerializationJSON) {
  test_linear_separable<float, TModelLogReg<float> >([](const SBaseArrayFloat2dPtr &features,
                                           const SArrayFloatPtr &labels) {
    return std::make_shared<TModelLogReg<float> >(features, labels, false, 1);
  });
}
//####################### LOG REG ############################################

//####################### POIS REG ###########################################
TEST(Model, PoisRegDoubleSerializationJSON) {
  test_linear_separable<double, TModelPoisReg<double> >(
      [](const SBaseArrayDouble2dPtr &features, const SArrayDoublePtr &labels) {
        return std::make_shared<TModelPoisReg<double> >(
            features, labels, LinkType::identity, false, 1);
      });
}
TEST(Model, PoisRegFloatSerializationJSON) {
  test_linear_separable<float, TModelPoisReg<float> >(
      [](const SBaseArrayFloat2dPtr &features, const SArrayFloatPtr &labels) {
        return std::make_shared<TModelPoisReg<float> >(
            features, labels, LinkType::identity, false, 1);
      });
}
//####################### POIS REG ###########################################

//####################### QUAD HINGE #########################################
TEST(Model, QuadHingeDoubleSerializationJSON) {
  test_linear_separable<double, TModelQuadraticHinge<double> >(
      [](const SBaseArrayDouble2dPtr &features, const SArrayDoublePtr &labels) {
        return std::make_shared<TModelQuadraticHinge<double> >(features, labels,
                                                               false);
      });
}
TEST(Model, QuadHingeFloatSerializationJSON) {
  test_linear_separable<float, TModelQuadraticHinge<float> >(
      [](const SBaseArrayFloat2dPtr &features, const SArrayFloatPtr &labels) {
        return std::make_shared<TModelQuadraticHinge<float> >(features, labels,
                                                              false);
      });
}
//####################### QUAD HINGE #########################################

//####################### HINGE ##############################################
TEST(Model, HingeDoubleSerializationJSON) {
  test_linear_separable<double, TModelHinge<double> >(
      [](const SBaseArrayDouble2dPtr &features, const SArrayDoublePtr &labels) {
        return std::make_shared<TModelHinge<double> >(features, labels, false);
      });
}
TEST(Model, HingeFloatSerializationJSON) {
  test_linear_separable<float, TModelHinge<float> >(
      [](const SBaseArrayFloat2dPtr &features, const SArrayFloatPtr &labels) {
        return std::make_shared<TModelHinge<float> >(features, labels, false);
      });
}
//####################### HINGE ##############################################

//####################### SMOOTH HINGE #######################################
TEST(Model, SmoothHingeDoubleSerializationJSON) {
  test_linear_separable<double, TModelSmoothedHinge<double> >(
      [](const SBaseArrayDouble2dPtr &features, const SArrayDoublePtr &labels) {
        return std::make_shared<TModelSmoothedHinge<double> >(features, labels,
                                                              false);
      });
}
TEST(Model, SmoothHingeFloatSerializationJSON) {
  test_linear_separable<float, TModelSmoothedHinge<float> >(
      [](const SBaseArrayFloat2dPtr &features, const SArrayFloatPtr &labels) {
        return std::make_shared<TModelSmoothedHinge<float> >(features, labels,
                                                             false);
      });
}
//####################### SMOOTH HINGE #######################################

//####################### ABSOLUTE REG #######################################
TEST(Model, AbsoluteRegressionDoubleSerializationJSON) {
  test_linear_separable<double, TModelAbsoluteRegression<double> >(
      [](const SBaseArrayDouble2dPtr &features, const SArrayDoublePtr &labels) {
        return std::make_shared<TModelAbsoluteRegression<double> >(features, labels,
                                                              false);
      });
}
TEST(Model, AbsoluteRegressionFloatSerializationJSON) {
  test_linear_separable<float, TModelAbsoluteRegression<float> >(
      [](const SBaseArrayFloat2dPtr &features, const SArrayFloatPtr &labels) {
        return std::make_shared<TModelAbsoluteRegression<float> >(features, labels,
                                                             false);
      });
}
//####################### ABSOLUTE REG #######################################

//####################### EPSIL INSENS #######################################
TEST(Model, EpsilonInsensitiveDoubleSerializationJSON) {
  test_linear_separable<double, TModelEpsilonInsensitive<double> >(
      [](const SBaseArrayDouble2dPtr &features, const SArrayDoublePtr &labels) {
        return std::make_shared<TModelEpsilonInsensitive<double> >(features, labels,
                                                              false, 100);
      });
}
TEST(Model, EpsilonInsensitiveFloatSerializationJSON) {
  test_linear_separable<float, TModelEpsilonInsensitive<float> >(
      [](const SBaseArrayFloat2dPtr &features, const SArrayFloatPtr &labels) {
        return std::make_shared<TModelEpsilonInsensitive<float> >(features, labels,
                                                             false, 100);
      });
}
//####################### EPSIL INSENS #######################################

//####################### GEN LIN WITH INT ###################################
TEST(Model, GeneralizedLinearDoubleSerializationJSON) {
  test_linear_separable<double, TModelGeneralizedLinearWithIntercepts<double> >(
      [](const SBaseArrayDouble2dPtr &features, const SArrayDoublePtr &labels) {
        return std::make_shared<TModelGeneralizedLinearWithIntercepts<double> >(features, labels,
                                                              false);
      });
}
TEST(Model, GeneralizedLinearSerializationJSON) {
  test_linear_separable<float, TModelGeneralizedLinearWithIntercepts<float> >(
      [](const SBaseArrayFloat2dPtr &features, const SArrayFloatPtr &labels) {
        return std::make_shared<TModelGeneralizedLinearWithIntercepts<float> >(features, labels,
                                                             false);
      });
}
//####################### GEN LIN WITH INT ###################################

//####################### HUBER ##############################################
TEST(Model, HuberDoubleSerializationJSON) {
  test_linear_separable<double, TModelHuber<double> >(
      [](const SBaseArrayDouble2dPtr &features, const SArrayDoublePtr &labels) {
        return std::make_shared<TModelHuber<double> >(features, labels,
                                                              false, 100);
      });
}
TEST(Model, HuberFloatSerializationJSON) {
  test_linear_separable<float, TModelHuber<float> >(
      [](const SBaseArrayFloat2dPtr &features, const SArrayFloatPtr &labels) {
        return std::make_shared<TModelHuber<float> >(features, labels,
                                                             false, 100);
      });
}
//####################### HUBER ##############################################

//####################### LIN REG INT ########################################
TEST(Model, LinRegWithInterceptsDoubleSerializationJSON) {
  test_linear_separable<double, TModelLinRegWithIntercepts<double> >(
      [](const SBaseArrayDouble2dPtr &features, const SArrayDoublePtr &labels) {
        return std::make_shared<TModelLinRegWithIntercepts<double> >(features, labels,
                                                              false);
      });
}
TEST(Model, LinRegWithInterceptsFloatSerializationJSON) {
  test_linear_separable<float, TModelLinRegWithIntercepts<float> >(
      [](const SBaseArrayFloat2dPtr &features, const SArrayFloatPtr &labels) {
        return std::make_shared<TModelLinRegWithIntercepts<float> >(features, labels,
                                                             false);
      });
}
//####################### LIN REG INT ########################################

//####################### MODIFIED HUBER #####################################
TEST(Model, ModifiedHuberDoubleSerializationJSON) {
  test_linear_separable<double, TModelModifiedHuber<double> >(
      [](const SBaseArrayDouble2dPtr &features, const SArrayDoublePtr &labels) {
        return std::make_shared<TModelModifiedHuber<double> >(features, labels,
                                                              false);
      });
}
TEST(Model, ModifiedHuberSerializationJSON) {
  test_linear_separable<float, TModelModifiedHuber<float> >(
      [](const SBaseArrayFloat2dPtr &features, const SArrayFloatPtr &labels) {
        return std::make_shared<TModelModifiedHuber<float> >(features, labels,
                                                             false);
      });
}
//####################### MODIFIED HUBER #####################################

#ifdef ADD_MAIN
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
