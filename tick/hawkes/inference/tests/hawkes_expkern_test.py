# License: BSD 3 clause

import unittest

import numpy as np

from tick.base.inference import InferenceTest
from tick.hawkes import SimuHawkesExpKernels
from tick.hawkes.inference import HawkesExpKern
from tick.hawkes.model.model_hawkes_expkern_leastsq import \
    ModelHawkesExpKernLeastSq
from tick.hawkes.model.model_hawkes_expkern_loglik import \
    ModelHawkesExpKernLogLik
from tick.prox import ProxNuclear
from tick.prox import ProxPositive, ProxL1, ProxL2Sq, ProxElasticNet
from tick.solver import AGD, GD, BFGS, SGD, SVRG

solvers = ['gd', 'agd', 'svrg', 'bfgs', 'sgd']
penalties = ['none', 'l2', 'l1', 'nuclear', 'elasticnet']
gofits = ['least-squares', 'likelihood']


class Test(InferenceTest):
    def setUp(self):
        np.random.seed(13069)
        self.float_1 = 5.23e-4
        self.float_2 = 3.86e-2
        self.int_1 = 3198
        self.int_2 = 230

        self.events = [np.cumsum(np.random.rand(10 + i)) for i in range(3)]

        self.decays = 3.

    @staticmethod
    def get_train_data(n_nodes=3, betas=1.):
        np.random.seed(130947)
        baseline = np.random.rand(n_nodes)
        adjacency = np.random.rand(n_nodes, n_nodes)
        if isinstance(betas, (int, float)):
            betas = np.ones((n_nodes, n_nodes)) * betas

        sim = SimuHawkesExpKernels(adjacency=adjacency, decays=betas,
                                   baseline=baseline, verbose=False,
                                   seed=13487, end_time=3000)
        sim.adjust_spectral_radius(0.8)
        adjacency = sim.adjacency
        sim.simulate()

        return sim.timestamps, baseline, adjacency

    @staticmethod
    def estimation_error(estimated, original):
        return np.linalg.norm(original - estimated) ** 2 / \
               np.linalg.norm(original) ** 2

    def test_HawkesExpKern_fit(self):
        """...Test HawkesExpKern fit with different solvers
        and penalties
        """
        sto_seed = 179312
        n_nodes = 2
        events, baseline, adjacency = Test.get_train_data(
            n_nodes=n_nodes, betas=self.decays)
        start = 0.3
        initial_adjacency_error = \
            Test.estimation_error(start * np.ones((n_nodes, n_nodes)),
                                  adjacency)

        for gofit in gofits:
            for penalty in penalties:
                for solver in solvers:

                    solver_kwargs = {
                        'penalty': penalty,
                        'tol': 1e-10,
                        'solver': solver,
                        'verbose': False,
                        'max_iter': 10,
                        'gofit': gofit
                    }

                    if penalty != 'none':
                        solver_kwargs['C'] = 50

                    if solver in ['sgd', 'svrg']:
                        solver_kwargs['random_state'] = sto_seed

                    # manually set step
                    if solver == 'sgd' and gofit == 'likelihood':
                        solver_kwargs['step'] = 3e-1
                    elif solver == 'sgd' and gofit == 'least-squares':
                        solver_kwargs['step'] = 1e-5
                    elif solver == 'svrg' and gofit == 'likelihood':
                        solver_kwargs['step'] = 1e-3
                    elif solver == 'svrg' and gofit == 'least-squares':
                        continue

                    if solver == 'bfgs':
                        # BFGS only accepts ProxZero and ProxL2sq for now
                        if penalty != 'l2':
                            continue

                    if penalty == 'nuclear':
                        # Nuclear penalty only compatible with batch solvers
                        if solver in \
                                HawkesExpKern._solvers_stochastic:
                            continue

                    learner = HawkesExpKern(self.decays, **solver_kwargs)
                    learner.fit(events, start=start)
                    adjacency_error = Test.estimation_error(
                        learner.adjacency, adjacency)
                    self.assertLess(
                        adjacency_error, initial_adjacency_error * 0.8,
                        "solver %s with penalty %s and "
                        "gofit %s reached too high "
                        "baseline error" % (solver, penalty, gofit))

    def test_HawkesExpKern_fit_start(self):
        """...Test HawkesExpKern starting point of fit method
        """
        n_nodes = len(self.events)
        n_coefs = n_nodes + n_nodes * n_nodes
        # Do not step
        learner = HawkesExpKern(self.decays, max_iter=-1)

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

    def test_HawkesExpKern_score(self):
        """...Test HawkesExpKern score method
        """
        n_nodes = 2
        n_realizations = 3

        train_events = [[
            np.cumsum(np.random.rand(4 + i)) for i in range(n_nodes)
        ] for _ in range(n_realizations)]

        test_events = [[
            np.cumsum(np.random.rand(4 + i)) for i in range(n_nodes)
        ] for _ in range(n_realizations)]

        learner = HawkesExpKern(self.decays, record_every=1)

        msg = '^You must either call `fit` before `score` or provide events$'
        with self.assertRaisesRegex(ValueError, msg):
            learner.score()

        given_baseline = np.random.rand(n_nodes)
        given_adjacency = np.random.rand(n_nodes, n_nodes)

        learner.fit(train_events)

        train_score_current_coeffs = learner.score()
        self.assertAlmostEqual(train_score_current_coeffs, 2.0855840)

        train_score_given_coeffs = learner.score(baseline=given_baseline,
                                                 adjacency=given_adjacency)
        self.assertAlmostEqual(train_score_given_coeffs, 0.59502417)

        test_score_current_coeffs = learner.score(test_events)
        self.assertAlmostEqual(test_score_current_coeffs, 1.6001762)

        test_score_given_coeffs = learner.score(
            test_events, baseline=given_baseline, adjacency=given_adjacency)
        self.assertAlmostEqual(test_score_given_coeffs, 0.89322199)

    @staticmethod
    def specific_solver_kwargs(solver):
        """...A simple method to as systematically some kwargs to our tests
        """
        return dict()

    def test_HawkesExpKern_settings(self):
        """...Test HawkesExpKern basic settings
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
            learner = HawkesExpKern(self.decays, solver=solver,
                                    **Test.specific_solver_kwargs(solver))

            solver_class = solver_class_map[solver]
            self.assertTrue(isinstance(learner._solver_obj, solver_class))

            msg = "solver is readonly in HawkesExpKern"
            with self.assertRaisesRegex(AttributeError, msg):
                learner.solver = solver

        msg = '^``solver`` must be one of agd, bfgs, gd, sgd, ' \
              'svrg, got wrong_name$'
        with self.assertRaisesRegex(ValueError, msg):
            HawkesExpKern(self.decays, solver='wrong_name')

        # prox
        prox_class_map = {
            'none': ProxPositive,
            'l1': ProxL1,
            'l2': ProxL2Sq,
            'elasticnet': ProxElasticNet,
            'nuclear': ProxNuclear
        }
        for penalty in penalties:
            learner = HawkesExpKern(self.decays, penalty=penalty)
            prox_class = prox_class_map[penalty]
            self.assertTrue(isinstance(learner._prox_obj, prox_class))

            msg = "penalty is readonly in HawkesExpKern"
            with self.assertRaisesRegex(AttributeError, msg):
                learner.penalty = penalty

        msg = '^``penalty`` must be one of elasticnet, l1, l2, none, ' \
              'nuclear, got wrong_name$'
        with self.assertRaisesRegex(ValueError, msg):
            HawkesExpKern(self.decays, penalty='wrong_name')

        # models
        model_class_map = {
            'least-squares': ModelHawkesExpKernLeastSq,
            'likelihood': ModelHawkesExpKernLogLik
        }
        for gofit in gofits:
            learner = HawkesExpKern(self.decays, gofit=gofit)
            model_class = model_class_map[gofit]
            self.assertTrue(isinstance(learner._model_obj, model_class))

            msg = "gofit is readonly in HawkesExpKern"
            with self.assertRaisesRegex(AttributeError, msg):
                learner.gofit = gofit

        msg = "^Parameter gofit \(goodness of fit\) must be either " \
              "'least-squares' or 'likelihood'$"
        with self.assertRaisesRegex(ValueError, msg):
            HawkesExpKern(self.decays, gofit='wrong_name')

    def test_HawkesExpKern_model_settings(self):
        """...Test HawkesExpKern setting of parameters of model
        """
        n_nodes = len(self.events)
        decay_array = np.random.rand(n_nodes, n_nodes)

        for gofit in gofits:
            learner = HawkesExpKern(self.float_1, gofit=gofit)
            self.assertEqual(learner.decays, self.float_1)
            self.assertEqual(learner._model_obj.decays, self.float_1)

            if gofit == "least-squares":
                learner = HawkesExpKern(decay_array, gofit=gofit)
                np.testing.assert_array_equal(learner.decays, decay_array)
                np.testing.assert_array_equal(learner._model_obj.decays,
                                              decay_array)

            else:
                msg = "With 'likelihood' goodness of fit, you must provide " \
                      "a constant decay for all kernels"
                with self.assertRaisesRegex(NotImplementedError, msg):
                    learner = HawkesExpKern(decay_array, gofit=gofit)

            msg = "decays is readonly in HawkesExpKern"
            with self.assertRaisesRegex(AttributeError, msg):
                learner.decays = self.float_2

    def test_HawkesExpKern_penalty_C(self):
        """...Test HawkesExpKern setting of parameter of C
        """

        for penalty in penalties:
            if penalty != 'none':
                learner = HawkesExpKern(self.decays, penalty=penalty,
                                        C=self.float_1)
                self.assertEqual(learner.C, self.float_1)
                self.assertEqual(learner._prox_obj.strength, 1. / self.float_1)
                learner.C = self.float_2
                self.assertEqual(learner.C, self.float_2)
                self.assertEqual(learner._prox_obj.strength, 1. / self.float_2)

                msg = '^``C`` must be positive, got -1$'
                with self.assertRaisesRegex(ValueError, msg):
                    HawkesExpKern(self.decays, penalty=penalty, C=-1)

            else:
                learner = HawkesExpKern(self.decays, penalty=penalty)
                msg = '^You cannot set C for penalty "%s"$' % penalty
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner.C = self.float_1

            msg = '^``C`` must be positive, got -2$'
            with self.assertRaisesRegex(ValueError, msg):
                learner.C = -2

    def test_HawkesExpKern_penalty_elastic_net_ratio(self):
        """...Test HawkesExpKern setting of parameter of
        elastic_net_ratio
        """
        ratio_1 = 0.6
        ratio_2 = 0.3

        for penalty in penalties:
            if penalty == 'elasticnet':

                learner = HawkesExpKern(self.decays, penalty=penalty,
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
                learner = HawkesExpKern(self.decays, penalty=penalty)
                msg = '^Penalty "%s" has no elastic_net_ratio attribute$$' % \
                      penalty
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner.elastic_net_ratio = ratio_1

    def test_HawkesExpKern_solver_basic_settings(self):
        """...Test HawkesExpKern setting of basic parameters
        of solver
        """
        for solver in solvers:
            # tol
            learner = HawkesExpKern(self.decays, solver=solver,
                                    tol=self.float_1,
                                    **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.tol, self.float_1)
            self.assertEqual(learner._solver_obj.tol, self.float_1)
            learner.tol = self.float_2
            self.assertEqual(learner.tol, self.float_2)
            self.assertEqual(learner._solver_obj.tol, self.float_2)

            # max_iter
            learner = HawkesExpKern(self.decays, solver=solver,
                                    max_iter=self.int_1,
                                    **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.max_iter, self.int_1)
            self.assertEqual(learner._solver_obj.max_iter, self.int_1)
            learner.max_iter = self.int_2
            self.assertEqual(learner.max_iter, self.int_2)
            self.assertEqual(learner._solver_obj.max_iter, self.int_2)

            # verbose
            learner = HawkesExpKern(self.decays, solver=solver, verbose=True,
                                    **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.verbose, True)
            self.assertEqual(learner._solver_obj.verbose, True)
            learner.verbose = False
            self.assertEqual(learner.verbose, False)
            self.assertEqual(learner._solver_obj.verbose, False)

            learner = HawkesExpKern(self.decays, solver=solver, verbose=False,
                                    **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.verbose, False)
            self.assertEqual(learner._solver_obj.verbose, False)
            learner.verbose = True
            self.assertEqual(learner.verbose, True)
            self.assertEqual(learner._solver_obj.verbose, True)

            # print_every
            learner = HawkesExpKern(self.decays, solver=solver,
                                    print_every=self.int_1,
                                    **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.print_every, self.int_1)
            self.assertEqual(learner._solver_obj.print_every, self.int_1)
            learner.print_every = self.int_2
            self.assertEqual(learner.print_every, self.int_2)
            self.assertEqual(learner._solver_obj.print_every, self.int_2)

            # record_every
            learner = HawkesExpKern(self.decays, solver=solver,
                                    record_every=self.int_1,
                                    **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.record_every, self.int_1)
            self.assertEqual(learner._solver_obj.record_every, self.int_1)
            learner.record_every = self.int_2
            self.assertEqual(learner.record_every, self.int_2)
            self.assertEqual(learner._solver_obj.record_every, self.int_2)

    def test_HawkesExpKern_solver_step(self):
        """...Test HawkesExpKern setting of step parameter
        of solver
        """
        for solver in solvers:
            if solver in ['bfgs']:
                msg = '^Solver "%s" has no settable step$' % solver
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner = HawkesExpKern(
                        self.decays, solver=solver, step=1,
                        **Test.specific_solver_kwargs(solver))
                    self.assertIsNone(learner.step)
            else:
                learner = HawkesExpKern(self.decays, solver=solver,
                                        step=self.float_1,
                                        **Test.specific_solver_kwargs(solver))
                self.assertEqual(learner.step, self.float_1)
                self.assertEqual(learner._solver_obj.step, self.float_1)
                learner.step = self.float_2
                self.assertEqual(learner.step, self.float_2)
                self.assertEqual(learner._solver_obj.step, self.float_2)

            if solver in ['sgd']:
                msg = '^SGD step needs to be tuned manually$'
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner = HawkesExpKern(self.decays, solver='sgd',
                                            max_iter=1)
                    learner.fit(self.events, 0.3)

    def test_HawkesExpKern_solver_random_state(self):
        """...Test HawkesExpKern setting of random_state
        parameter of solver
        """
        for solver in solvers:
            if solver in ['bfgs', 'agd', 'gd']:
                msg = '^Solver "%s" has no settable random_state$' % solver
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner = HawkesExpKern(
                        self.decays, solver=solver, random_state=1,
                        **Test.specific_solver_kwargs(solver))
                    self.assertIsNone(learner.random_state)
            else:
                learner = HawkesExpKern(self.decays, solver=solver,
                                        random_state=self.int_1,
                                        **Test.specific_solver_kwargs(solver))
                self.assertEqual(learner.random_state, self.int_1)
                self.assertEqual(learner._solver_obj.seed, self.int_1)

                msg = '^random_state must be positive, got -1$'
                with self.assertRaisesRegex(ValueError, msg):
                    HawkesExpKern(self.decays, solver=solver, random_state=-1)

            msg = '^random_state is readonly in HawkesExpKern'
            with self.assertRaisesRegex(AttributeError, msg):
                learner = HawkesExpKern(self.decays, solver=solver,
                                        **Test.specific_solver_kwargs(solver))
                learner.random_state = self.int_2

    def test_corresponding_simu(self):
        """...Test that the corresponding simulation object is correctly
        built
        """
        learner = HawkesExpKern(self.decays, max_iter=10)
        learner.fit(self.events)

        corresponding_simu = learner._corresponding_simu()
        self.assertEqual(corresponding_simu.decays, learner.decays)
        np.testing.assert_array_equal(corresponding_simu.baseline,
                                      learner.baseline)
        np.testing.assert_array_equal(corresponding_simu.adjacency,
                                      learner.adjacency)


if __name__ == "__main__":
    unittest.main()
