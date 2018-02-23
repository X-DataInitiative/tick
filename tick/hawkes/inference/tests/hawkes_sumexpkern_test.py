# License: BSD 3 clause

import unittest

import numpy as np

from tick.base.inference import InferenceTest
from tick.hawkes import SimuHawkesSumExpKernels, SimuHawkesMulti
from tick.hawkes.inference import HawkesSumExpKern
from tick.prox import ProxNuclear
from tick.prox import ProxPositive, ProxL1, ProxL2Sq, ProxElasticNet
from tick.solver import AGD, GD, BFGS, SGD, SVRG

solvers = ['gd', 'agd', 'svrg', 'bfgs', 'sgd']
penalties = ['none', 'l2', 'l1', 'elasticnet']


class Test(InferenceTest):
    def setUp(self):
        np.random.seed(13069)
        self.float_1 = 5.23e-4
        self.float_2 = 3.86e-2
        self.int_1 = 3198
        self.int_2 = 230

        self.events = [np.cumsum(np.random.rand(10 + i)) for i in range(3)]

        self.n_decays = 2
        self.decays = np.random.rand(self.n_decays)

    @staticmethod
    def get_train_data(decays, n_nodes=3, n_decays=2):
        np.random.seed(130947)
        baseline = np.random.rand(n_nodes)
        adjacency = np.random.rand(n_nodes, n_nodes, n_decays)

        sim = SimuHawkesSumExpKernels(adjacency=adjacency, decays=decays,
                                      baseline=baseline, verbose=False,
                                      seed=13487, end_time=3000)
        sim.adjust_spectral_radius(0.8)
        adjacency = sim.adjacency
        sim.simulate()

        return sim.timestamps, baseline, adjacency

    def simulate_sparse_realization(self):
        """Simulate realization in which some nodes are sometimes empty
        """
        baseline = np.array([0.3, 0.001])
        adjacency = np.array([[[0.5, 0.3], [0.8, 0.2]], [[0., 0.], [1.3,
                                                                    0.9]]])

        sim = SimuHawkesSumExpKernels(adjacency=adjacency, decays=self.decays,
                                      baseline=baseline, verbose=False,
                                      seed=13487, end_time=500)
        sim.adjust_spectral_radius(0.8)
        multi = SimuHawkesMulti(sim, n_simulations=100)

        adjacency = sim.adjacency
        multi.simulate()

        # Check that some but not all realizations are empty
        self.assertGreater(max(map(lambda r: len(r[1]), multi.timestamps)), 1)
        self.assertEqual(min(map(lambda r: len(r[1]), multi.timestamps)), 0)

        return baseline, adjacency, multi.timestamps

    def test_sparse(self):
        """...Test that original coeffs are correctly retrieved when some
        realizations are empty
        """
        baseline, adjacency, events = self.simulate_sparse_realization()

        learner = HawkesSumExpKern(self.decays, verbose=False)
        learner.fit(events)

        np.testing.assert_array_almost_equal(learner.baseline, baseline,
                                             decimal=1)
        np.testing.assert_array_almost_equal(learner.adjacency, adjacency,
                                             decimal=1)

    @staticmethod
    def estimation_error(estimated, original):
        return np.linalg.norm(original - estimated) ** 2 / \
               np.linalg.norm(original) ** 2

    def test_HawkesSumExpKern_fit(self):
        """...Test HawkesSumExpKern fit with different solvers
        and penalties
        """
        sto_seed = 179312
        n_nodes = 2
        events, baseline, adjacency = Test.get_train_data(
            self.decays, n_nodes=n_nodes, n_decays=self.n_decays)
        start = 0.01
        initial_adjacency_error = \
            Test.estimation_error(start * np.ones((n_nodes, n_nodes)),
                                  adjacency)

        for penalty in penalties:
            for solver in solvers:

                solver_kwargs = {
                    'penalty': penalty,
                    'tol': 1e-10,
                    'solver': solver,
                    'verbose': False,
                    'max_iter': 1000
                }

                if penalty != 'none':
                    solver_kwargs['C'] = 50

                if solver in ['sgd', 'svrg']:
                    solver_kwargs['random_state'] = sto_seed

                # manually set step
                if solver == 'sgd':
                    solver_kwargs['step'] = 1e-5
                elif solver == 'svrg':
                    continue

                if solver == 'bfgs':
                    # BFGS only accepts ProxZero and ProxL2sq for now
                    if penalty != 'l2':
                        continue

                if penalty == 'nuclear':
                    # Nuclear penalty only compatible with batch solvers
                    if solver in \
                            HawkesSumExpKern._solvers_stochastic:
                        continue

                learner = HawkesSumExpKern(self.decays, **solver_kwargs)
                learner.fit(events, start=start)
                adjacency_error = Test.estimation_error(
                    learner.adjacency, adjacency)
                self.assertLess(
                    adjacency_error, initial_adjacency_error * 0.8,
                    "solver %s with penalty %s "
                    "reached too high baseline error" % (solver, penalty))

    def test_HawkesSumExpKern_score(self):
        """...Test HawkesSumExpKern score method
        """
        n_nodes = 2
        n_realizations = 3

        train_events = [[
            np.cumsum(np.random.rand(4 + i)) for i in range(n_nodes)
        ] for _ in range(n_realizations)]

        test_events = [[
            np.cumsum(np.random.rand(4 + i)) for i in range(n_nodes)
        ] for _ in range(n_realizations)]

        learner = HawkesSumExpKern(self.decays, record_every=1)

        msg = '^You must either call `fit` before `score` or provide events$'
        with self.assertRaisesRegex(ValueError, msg):
            learner.score()

        given_baseline = np.random.rand(n_nodes)
        given_adjacency = np.random.rand(n_nodes, n_nodes, self.n_decays)

        learner.fit(train_events)

        train_score_current_coeffs = learner.score()
        self.assertAlmostEqual(train_score_current_coeffs, 1.684827141)

        train_score_given_coeffs = learner.score(baseline=given_baseline,
                                                 adjacency=given_adjacency)
        self.assertAlmostEqual(train_score_given_coeffs, 1.16247892)

        test_score_current_coeffs = learner.score(test_events)
        self.assertAlmostEqual(test_score_current_coeffs, 1.66494295)

        test_score_given_coeffs = learner.score(
            test_events, baseline=given_baseline, adjacency=given_adjacency)
        self.assertAlmostEqual(test_score_given_coeffs, 1.1081362)

    def test_HawkesSumExpKern_fit_start(self):
        """...Test HawkesSumExpKern starting point of fit method
        """
        n_nodes = len(self.events)
        n_coefs = n_nodes + n_nodes * n_nodes * self.n_decays
        # Do not step
        learner = HawkesSumExpKern(self.decays, max_iter=-1)

        learner.fit(self.events)
        np.testing.assert_array_equal(learner.coeffs, np.ones(n_coefs))

        learner.fit(self.events, start=self.float_1)
        np.testing.assert_array_equal(learner.coeffs,
                                      np.ones(n_coefs) * self.float_1)

        learner.fit(self.events, start=self.int_1)
        np.testing.assert_array_equal(learner.coeffs,
                                      np.ones(n_coefs) * self.int_1)

        random_coeffs = np.random.rand(n_coefs)
        learner.fit(self.events, start=random_coeffs)
        np.testing.assert_array_equal(learner.coeffs, random_coeffs)

    @staticmethod
    def specific_solver_kwargs(solver):
        """...A simple method to as systematically some kwargs to our tests
        """
        return dict()

    def test_HawkesSumExpKern_settings(self):
        """...Test HawkesSumExpKern basic settings
        """
        # solver
        solver_class_map = {
            'gd': GD,
            'agd': AGD,
            'sgd': SGD,
            'svrg': SVRG,
            'bfgs': BFGS
        }
        for solver in solvers:
            learner = HawkesSumExpKern(self.decays, solver=solver,
                                       **Test.specific_solver_kwargs(solver))

            solver_class = solver_class_map[solver]
            self.assertTrue(isinstance(learner._solver_obj, solver_class))

            msg = "solver is readonly in HawkesSumExpKern"
            with self.assertRaisesRegex(AttributeError, msg):
                learner.solver = solver

        msg = '^``solver`` must be one of agd, bfgs, gd, sgd, ' \
              'svrg, got wrong_name$'
        with self.assertRaisesRegex(ValueError, msg):
            HawkesSumExpKern(self.decays, solver='wrong_name')

        # prox
        prox_class_map = {
            'none': ProxPositive,
            'l1': ProxL1,
            'l2': ProxL2Sq,
            'elasticnet': ProxElasticNet,
            'nuclear': ProxNuclear
        }
        for penalty in penalties:
            learner = HawkesSumExpKern(self.decays, penalty=penalty)
            prox_class = prox_class_map[penalty]
            self.assertTrue(isinstance(learner._prox_obj, prox_class))

            msg = "penalty is readonly in HawkesSumExpKern"
            with self.assertRaisesRegex(AttributeError, msg):
                learner.penalty = penalty

        msg = '^``penalty`` must be one of elasticnet, l1, l2, none, ' \
              'got wrong_name$'
        with self.assertRaisesRegex(ValueError, msg):
            HawkesSumExpKern(self.decays, penalty='wrong_name')

    def test_HawkesSumExpKern_model_settings(self):
        """...Test HawkesSumExpKern setting of parameters
        of model
        """
        # Single baseline
        learner = HawkesSumExpKern(self.decays)
        np.testing.assert_array_equal(learner.decays, self.decays)
        np.testing.assert_array_equal(learner._model_obj.decays, self.decays)
        self.assertEqual(learner.n_baselines, 1)
        self.assertIsNone(learner.period_length)

        # Multiple baselines
        n_baselines = 3
        period_length = 2.
        learner = HawkesSumExpKern(self.decays, n_baselines=n_baselines,
                                   period_length=period_length)
        self.assertEqual(learner.n_baselines, n_baselines)
        self.assertEqual(learner._model_obj.n_baselines, n_baselines)
        self.assertEqual(learner.period_length, period_length)
        self.assertEqual(learner._model_obj.period_length, period_length)

        msg = "decays is readonly in HawkesSumExpKern"
        with self.assertRaisesRegex(AttributeError, msg):
            learner.decays = self.decays + 1
        msg = "n_baselines is readonly in HawkesSumExpKern"
        with self.assertRaisesRegex(AttributeError, msg):
            learner.n_baselines = n_baselines + 1
        msg = "period_length is readonly in HawkesSumExpKern"
        with self.assertRaisesRegex(AttributeError, msg):
            learner.period_length = period_length + 1

        msg = "You must fit data before getting estimated baseline"
        with self.assertRaisesRegex(ValueError, msg):
            learner.baseline
        msg = "You must fit data before getting estimated adjacency"
        with self.assertRaisesRegex(ValueError, msg):
            learner.adjacency

    def test_HawkesSumExpKern_penalty_C(self):
        """...Test HawkesSumExpKern setting of parameter of C
        """
        for penalty in penalties:
            if penalty != 'none':
                learner = HawkesSumExpKern(self.decays, penalty=penalty,
                                           C=self.float_1)
                self.assertEqual(learner.C, self.float_1)
                self.assertEqual(learner._prox_obj.strength, 1. / self.float_1)
                learner.C = self.float_2
                self.assertEqual(learner.C, self.float_2)
                self.assertEqual(learner._prox_obj.strength, 1. / self.float_2)

                msg = '^``C`` must be positive, got -1$'
                with self.assertRaisesRegex(ValueError, msg):
                    HawkesSumExpKern(self.decays, penalty=penalty, C=-1)

            else:
                learner = HawkesSumExpKern(self.decays, penalty=penalty)
                msg = '^You cannot set C for penalty "%s"$' % penalty
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner.C = self.float_1

            msg = '^``C`` must be positive, got -2$'
            with self.assertRaisesRegex(ValueError, msg):
                learner.C = -2

    def test_HawkesSumExpKern_penalty_elastic_net_ratio(self):
        """...Test HawkesSumExpKern setting of parameter of
        elastic_net_ratio
        """
        ratio_1 = 0.6
        ratio_2 = 0.3

        for penalty in penalties:
            if penalty == 'elasticnet':

                learner = HawkesSumExpKern(self.decays, penalty=penalty,
                                           C=self.float_1,
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
                learner = HawkesSumExpKern(self.decays, penalty=penalty)
                msg = '^Penalty "%s" has no elastic_net_ratio attribute$$' % \
                      penalty
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner.elastic_net_ratio = ratio_1

    def test_HawkesSumExpKern_solver_basic_settings(self):
        """...Test HawkesSumExpKern setting of basic parameters
        of solver
        """
        for solver in solvers:
            # tol
            learner = HawkesSumExpKern(self.decays, solver=solver,
                                       tol=self.float_1,
                                       **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.tol, self.float_1)
            self.assertEqual(learner._solver_obj.tol, self.float_1)
            learner.tol = self.float_2
            self.assertEqual(learner.tol, self.float_2)
            self.assertEqual(learner._solver_obj.tol, self.float_2)

            # max_iter
            learner = HawkesSumExpKern(self.decays, solver=solver,
                                       max_iter=self.int_1,
                                       **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.max_iter, self.int_1)
            self.assertEqual(learner._solver_obj.max_iter, self.int_1)
            learner.max_iter = self.int_2
            self.assertEqual(learner.max_iter, self.int_2)
            self.assertEqual(learner._solver_obj.max_iter, self.int_2)

            # verbose
            learner = HawkesSumExpKern(self.decays, solver=solver,
                                       verbose=True,
                                       **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.verbose, True)
            self.assertEqual(learner._solver_obj.verbose, True)
            learner.verbose = False
            self.assertEqual(learner.verbose, False)
            self.assertEqual(learner._solver_obj.verbose, False)

            learner = HawkesSumExpKern(self.decays, solver=solver,
                                       verbose=False,
                                       **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.verbose, False)
            self.assertEqual(learner._solver_obj.verbose, False)
            learner.verbose = True
            self.assertEqual(learner.verbose, True)
            self.assertEqual(learner._solver_obj.verbose, True)

            # print_every
            learner = HawkesSumExpKern(self.decays, solver=solver,
                                       print_every=self.int_1,
                                       **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.print_every, self.int_1)
            self.assertEqual(learner._solver_obj.print_every, self.int_1)
            learner.print_every = self.int_2
            self.assertEqual(learner.print_every, self.int_2)
            self.assertEqual(learner._solver_obj.print_every, self.int_2)

            # record_every
            learner = HawkesSumExpKern(self.decays, solver=solver,
                                       record_every=self.int_1,
                                       **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.record_every, self.int_1)
            self.assertEqual(learner._solver_obj.record_every, self.int_1)
            learner.record_every = self.int_2
            self.assertEqual(learner.record_every, self.int_2)
            self.assertEqual(learner._solver_obj.record_every, self.int_2)

    def test_HawkesSumExpKern_solver_step(self):
        """...Test HawkesSumExpKern setting of step parameter
        of solver
        """
        for solver in solvers:
            if solver in ['bfgs']:
                msg = '^Solver "%s" has no settable step$' % solver
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner = HawkesSumExpKern(
                        self.decays, solver=solver, step=1,
                        **Test.specific_solver_kwargs(solver))
                    self.assertIsNone(learner.step)
            else:
                learner = HawkesSumExpKern(
                    self.decays, solver=solver, step=self.float_1,
                    **Test.specific_solver_kwargs(solver))
                self.assertEqual(learner.step, self.float_1)
                self.assertEqual(learner._solver_obj.step, self.float_1)
                learner.step = self.float_2
                self.assertEqual(learner.step, self.float_2)
                self.assertEqual(learner._solver_obj.step, self.float_2)

            if solver in ['sgd']:
                msg = '^SGD step needs to be tuned manually$'
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner = HawkesSumExpKern(self.decays, solver='sgd',
                                               max_iter=1)
                    learner.fit(self.events, 0.3)

    def test_HawkesSumExpKern_solver_random_state(self):
        """...Test HawkesSumExpKern setting of random_state
        parameter of solver
        """
        for solver in solvers:
            if solver in ['bfgs', 'gd', 'agd']:
                msg = '^Solver "%s" has no settable random_state$' % solver
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner = HawkesSumExpKern(
                        self.decays, solver=solver, random_state=1,
                        **Test.specific_solver_kwargs(solver))
                    self.assertIsNone(learner.random_state)
            else:
                learner = HawkesSumExpKern(
                    self.decays, solver=solver, random_state=self.int_1,
                    **Test.specific_solver_kwargs(solver))
                self.assertEqual(learner.random_state, self.int_1)
                self.assertEqual(learner._solver_obj.seed, self.int_1)

                msg = '^random_state must be positive, got -1$'
                with self.assertRaisesRegex(ValueError, msg):
                    HawkesSumExpKern(self.decays, solver=solver,
                                     random_state=-1)

            msg = '^random_state is readonly in ' \
                  'HawkesSumExpKern'
            with self.assertRaisesRegex(AttributeError, msg):
                learner = HawkesSumExpKern(
                    self.decays, solver=solver,
                    **Test.specific_solver_kwargs(solver))
                learner.random_state = self.int_2

    def test_corresponding_simu(self):
        """...Test that the corresponding simulation object is correctly
        built
        """
        learner = HawkesSumExpKern(self.decays, max_iter=10)
        learner.fit(self.events)

        corresponding_simu = learner._corresponding_simu()
        np.testing.assert_array_equal(corresponding_simu.decays,
                                      learner.decays)
        np.testing.assert_array_equal(corresponding_simu.baseline,
                                      learner.baseline)
        np.testing.assert_array_equal(corresponding_simu.adjacency,
                                      learner.adjacency)

        learner = HawkesSumExpKern(self.decays, n_baselines=3, period_length=1,
                                   max_iter=10)
        learner.fit(self.events)

        corresponding_simu = learner._corresponding_simu()
        np.testing.assert_array_equal(corresponding_simu.decays,
                                      learner.decays)
        np.testing.assert_array_equal(corresponding_simu.baseline,
                                      learner.baseline)
        np.testing.assert_array_equal(corresponding_simu.adjacency,
                                      learner.adjacency)


if __name__ == "__main__":
    unittest.main()
