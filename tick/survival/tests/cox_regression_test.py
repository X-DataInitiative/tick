# License: BSD 3 clause

import unittest

import numpy as np

from tick.base.inference import InferenceTest
from tick.survival import SimuCoxReg, CoxRegression
from tick.simulation import weights_sparse_gauss
from tick.preprocessing.features_binarizer import FeaturesBinarizer
from tick.prox import ProxZero, ProxL1, ProxL2Sq, ProxElasticNet, ProxTV, \
    ProxBinarsity
from tick.solver import GD, AGD


class Test(InferenceTest):
    solvers = ['gd', 'agd']

    penalties = ['none', 'l2', 'l1', 'elasticnet', 'tv', 'binarsity']

    float_1 = 5.23e-4
    float_2 = 3.86e-2
    int_1 = 3198
    int_2 = 230

    @staticmethod
    def get_train_data(n_features=10, n_samples=10000, nnz=3, seed=12):
        np.random.seed(seed)
        coeffs0 = weights_sparse_gauss(n_features, nnz=nnz)
        features, times, censoring = SimuCoxReg(coeffs0,
                                                verbose=False).simulate()
        return features, times, censoring

    def test_CoxRegression_fit(self):
        """...Test CoxRegression fit with different solvers and penalties
        """
        raw_features, times, censoring = Test.get_train_data()

        coeffs_pen = {
            'none':
                np.array([
                    -0.03068462, 0.03940001, 0.16758354, -0.24838003,
                    0.16940664, 0.9650363, -0.14818724, -0.0802245,
                    -1.52869811, 0.0414509
                ]),
            'l2':
                np.array([
                    -0.02403681, 0.03455527, 0.13470436, -0.21654892,
                    0.16637723, 0.83125941, -0.08555382, -0.12914753,
                    -1.35294435, 0.02312935
                ]),
            'l1':
                np.array([
                    0., 1.48439371e-02, 1.03806171e-01, -1.57313537e-01,
                    1.40448847e-01, 8.05306416e-01, -5.41296030e-02,
                    -1.07753576e-01, -1.37612207e+00, 6.43289248e-05
                ]),
            'elasticnet':
                np.array([
                    0., 0.01011823, 0.10530518, -0.16885214, 0.14373715,
                    0.82547312, -0.06122141, -0.09479487, -1.39454662,
                    0.00312597
                ]),
            'tv':
                np.array([
                    0.03017556, 0.03714465, 0.0385349, -0.10169967, 0.15783755,
                    0.64860815, -0.00617636, -0.22235137, -1.07938977,
                    -0.07181225
                ]),
            'binarsity':
                np.array([
                    0.03794176, -0.04473702, 0.00339763, 0.00339763,
                    -0.16493989, 0.05497996, 0.05497996, 0.05497996,
                    -0.08457476, -0.08457476, 0.0294825, 0.13966702,
                    0.10251257, 0.02550264, -0.07207419, -0.05594102,
                    -0.10018038, -0.10018038, 0.10018038, 0.10018038,
                    -0.47859686, -0.06685181, -0.00850803, 0.55395669,
                    0.00556327, -0.00185442, -0.00185442, -0.00185442,
                    0.26010429, 0.09752455, -0.17881442, -0.17881442, 0.932516,
                    0.32095387, -0.49766315, -0.75580671, 0.0593833,
                    -0.01433773, 0.01077109, -0.05581666
                ])
        }

        for penalty in self.penalties:

            if penalty == 'binarsity':
                # binarize features
                n_cuts = 3
                binarizer = FeaturesBinarizer(n_cuts=n_cuts)
                features = binarizer.fit_transform(raw_features)
            else:
                features = raw_features

            for solver in self.solvers:

                solver_kwargs = {
                    'penalty': penalty,
                    'tol': 0,
                    'solver': solver,
                    'verbose': False,
                    'max_iter': 10
                }

                if penalty != 'none':
                    solver_kwargs['C'] = 50

                if penalty == 'binarsity':
                    solver_kwargs['blocks_start'] = binarizer.blocks_start
                    solver_kwargs['blocks_length'] = binarizer.blocks_length

                learner = CoxRegression(**solver_kwargs)
                learner.fit(features, times, censoring)

                np.testing.assert_array_almost_equal(coeffs_pen[penalty],
                                                     learner.coeffs, decimal=1)

    def test_CoxRegression_warm_start(self):
        """...Test CoxRegression warm start
        """
        features, times, censoring = Test.get_train_data()

        for solver in self.solvers:
            solver_kwargs = {
                'solver': solver,
                'max_iter': 2,
                'warm_start': True,
                'tol': 0,
                'penalty': 'none'
            }
            learner = CoxRegression(**solver_kwargs)
            learner.fit(features, times, censoring)
            score_1 = learner.score()
            learner.fit(features, times, censoring)
            score_2 = learner.score()
            # Thanks to warm start the score should have decreased (no
            # penalization here)
            self.assertLess(score_2, score_1)

        for solver in self.solvers:
            solver_kwargs = {
                'solver': solver,
                'max_iter': 2,
                'warm_start': False,
                'tol': 0,
                'penalty': 'none'
            }
            learner = CoxRegression(**solver_kwargs)
            learner.fit(features, times, censoring)
            score_1 = learner.score()
            learner.fit(features, times, censoring)
            score_2 = learner.score()
            # No warm start here, so the scores should be the same
            self.assertAlmostEqual(score_2, score_1)

    def test_CoxRegression_settings(self):
        """...Test CoxRegression basic settings
        """
        # solver
        solver_class_map = {'gd': GD, 'agd': AGD}
        for solver in self.solvers:
            learner = CoxRegression(solver=solver)
            solver_class = solver_class_map[solver]
            self.assertTrue(isinstance(learner._solver_obj, solver_class))

        msg = '^``solver`` must be one of agd, gd, got wrong_name$'
        with self.assertRaisesRegex(ValueError, msg):
            CoxRegression(solver='wrong_name')

        # prox
        prox_class_map = {
            'none': ProxZero,
            'l1': ProxL1,
            'l2': ProxL2Sq,
            'elasticnet': ProxElasticNet,
            'tv': ProxTV,
            'binarsity': ProxBinarsity
        }

        for penalty in self.penalties:
            if penalty == 'binarsity':
                learner = CoxRegression(penalty=penalty, blocks_start=[0],
                                        blocks_length=[1])
            else:
                learner = CoxRegression(penalty=penalty)

            prox_class = prox_class_map[penalty]
            self.assertTrue(isinstance(learner._prox_obj, prox_class))

        msg = '^``penalty`` must be one of binarsity, elasticnet, l1, l2, none, ' \
              'tv, got wrong_name$'
        with self.assertRaisesRegex(ValueError, msg):
            CoxRegression(penalty='wrong_name')

    def test_CoxRegression_penalty_C(self):
        """...Test CoxRegression setting of parameter of C
        """

        for penalty in self.penalties:
            if penalty != 'none':
                if penalty == 'binarsity':
                    learner = CoxRegression(penalty=penalty, C=self.float_1,
                                            blocks_start=[0],
                                            blocks_length=[1])
                else:
                    learner = CoxRegression(penalty=penalty, C=self.float_1)
                self.assertEqual(learner.C, self.float_1)
                self.assertEqual(learner._prox_obj.strength, 1. / self.float_1)
                learner.C = self.float_2
                self.assertEqual(learner.C, self.float_2)
                self.assertEqual(learner._prox_obj.strength, 1. / self.float_2)

                msg = '^``C`` must be positive, got -1$'
                with self.assertRaisesRegex(ValueError, msg):
                    if penalty == 'binarsity':
                        CoxRegression(penalty=penalty, C=-1, blocks_start=[0],
                                      blocks_length=[1])
                    else:
                        CoxRegression(penalty=penalty, C=-1)

            else:
                msg = '^You cannot set C for penalty "%s"$' % penalty
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    CoxRegression(penalty=penalty, C=self.float_1)

                learner = CoxRegression(penalty=penalty)
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner.C = self.float_1

            msg = '^``C`` must be positive, got -2$'
            with self.assertRaisesRegex(ValueError, msg):
                learner.C = -2

    def test_CoxRegression_penalty_elastic_net_ratio(self):
        """...Test CoxRegression setting of parameter of elastic_net_ratio
        """
        ratio_1 = 0.6
        ratio_2 = 0.3

        for penalty in self.penalties:
            if penalty == 'elasticnet':

                learner = CoxRegression(penalty=penalty, C=self.float_1,
                                        elastic_net_ratio=ratio_1)
                self.assertEqual(learner.C, self.float_1)
                self.assertEqual(learner.elastic_net_ratio, ratio_1)
                self.assertEqual(learner._prox_obj.strength, 1. / self.float_1)
                self.assertEqual(learner._prox_obj.ratio, ratio_1)

                learner.elastic_net_ratio = ratio_2
                self.assertEqual(learner.C, self.float_1)
                self.assertEqual(learner.elastic_net_ratio, ratio_2)
                self.assertEqual(learner._prox_obj.ratio, ratio_2)

            else:
                msg = '^Penalty "%s" has no elastic_net_ratio attribute$' % \
                      penalty
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    if penalty == 'binarsity':
                        CoxRegression(penalty=penalty, elastic_net_ratio=0.8,
                                      blocks_start=[0], blocks_length=[1])
                    else:
                        CoxRegression(penalty=penalty, elastic_net_ratio=0.8)

                if penalty == 'binarsity':
                    learner = CoxRegression(penalty=penalty, blocks_start=[0],
                                            blocks_length=[1])
                else:
                    learner = CoxRegression(penalty=penalty)
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner.elastic_net_ratio = ratio_1

    def test_CoxRegression_solver_basic_settings(self):
        """...Test CoxRegression setting of basic parameters of solver
        """
        for solver in self.solvers:
            # tol
            learner = CoxRegression(solver=solver, tol=self.float_1)
            self.assertEqual(learner.tol, self.float_1)
            self.assertEqual(learner._solver_obj.tol, self.float_1)
            learner.tol = self.float_2
            self.assertEqual(learner.tol, self.float_2)
            self.assertEqual(learner._solver_obj.tol, self.float_2)

            # max_iter
            learner = CoxRegression(solver=solver, max_iter=self.int_1)
            self.assertEqual(learner.max_iter, self.int_1)
            self.assertEqual(learner._solver_obj.max_iter, self.int_1)
            learner.max_iter = self.int_2
            self.assertEqual(learner.max_iter, self.int_2)
            self.assertEqual(learner._solver_obj.max_iter, self.int_2)

            # verbose
            learner = CoxRegression(solver=solver, verbose=True)
            self.assertEqual(learner.verbose, True)
            self.assertEqual(learner._solver_obj.verbose, True)
            learner.verbose = False
            self.assertEqual(learner.verbose, False)
            self.assertEqual(learner._solver_obj.verbose, False)

            learner = CoxRegression(solver=solver, verbose=False)
            self.assertEqual(learner.verbose, False)
            self.assertEqual(learner._solver_obj.verbose, False)
            learner.verbose = True
            self.assertEqual(learner.verbose, True)
            self.assertEqual(learner._solver_obj.verbose, True)

            # print_every
            learner = CoxRegression(solver=solver, print_every=self.int_1)
            self.assertEqual(learner.print_every, self.int_1)
            self.assertEqual(learner._solver_obj.print_every, self.int_1)
            learner.print_every = self.int_2
            self.assertEqual(learner.print_every, self.int_2)
            self.assertEqual(learner._solver_obj.print_every, self.int_2)

            # record_every
            learner = CoxRegression(solver=solver, record_every=self.int_1)
            self.assertEqual(learner.record_every, self.int_1)
            self.assertEqual(learner._solver_obj.record_every, self.int_1)
            learner.record_every = self.int_2
            self.assertEqual(learner.record_every, self.int_2)
            self.assertEqual(learner._solver_obj.record_every, self.int_2)

    def test_CoxRegression_solver_step(self):
        """...Test CoxRegression setting of step parameter of solver
        """
        for solver in self.solvers:
            learner = CoxRegression(solver=solver, step=self.float_1)
            self.assertEqual(learner.step, self.float_1)
            self.assertEqual(learner._solver_obj.step, self.float_1)
            learner.step = self.float_2
            self.assertEqual(learner.step, self.float_2)
            self.assertEqual(learner._solver_obj.step, self.float_2)

    def test_CoxRegression_score(self):
        """...Test CoxRegression score
        """
        features, times, censoring = Test.get_train_data()
        learner = CoxRegression()
        learner.fit(features, times, censoring)
        self.assertAlmostEqual(learner.score(), 3.856303803547875)

        features, times, censoring = Test.get_train_data(seed=123)
        self.assertAlmostEqual(
            learner.score(features, times, censoring), 5.556509086276002)

        msg = '^You must fit the model first$'
        learner = CoxRegression()
        with self.assertRaisesRegex(RuntimeError, msg):
            learner.score()

        msg = '^Passed ``features`` is None$'
        learner = CoxRegression().fit(features, times, censoring)
        with self.assertRaisesRegex(ValueError, msg):
            learner.score(None, times, censoring)

        msg = '^Passed ``times`` is None$'
        learner = CoxRegression().fit(features, times, censoring)
        with self.assertRaisesRegex(ValueError, msg):
            learner.score(times, None, censoring)

        msg = '^Passed ``censoring`` is None$'
        learner = CoxRegression().fit(features, times, censoring)
        with self.assertRaisesRegex(ValueError, msg):
            learner.score(features, times, None)


if __name__ == "__main__":
    unittest.main()
