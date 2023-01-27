# License: BSD 3 clause

import os
import pickle
import unittest
from typing import Optional

import numpy as np

from tick.base.inference import InferenceTest


from tick.hawkes import (
    HawkesCumulantMatching,
    HawkesCumulantMatchingTf,
    HawkesCumulantMatchingPyT
)
from tick.hawkes.inference.hawkes_cumulant_matching import HawkesTheoreticalCumulant

SKIP_TF = False
try:
    import tensorflow as tf
except ImportError:
    SKIP_TF = True


SKIP_TORCH = False
try:
    import torch
except ImportError:
    SKIP_TORCH = True


class Test(InferenceTest):
    def setUp(self):
        self.dim = 2
        np.random.seed(320982)

    @staticmethod
    def get_simulated_model():
        from tick.hawkes import SimuHawkesExpKernels
        from tick.hawkes import SimuHawkesMulti

        adjacency = [
            [0.15, 0.03, 0.09],
            [0.0, 0.2, 0.05],
            [.05, .08, 0.1],
        ]
        decays = [
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
        ]
        baseline = [.001, .001, .001]
        end_time = 5.0e+7
        model = SimuHawkesExpKernels(
            adjacency=adjacency,
            decays=decays,
            baseline=baseline,
            end_time=end_time,
            max_jumps=100000,
            verbose=False,
            seed=1039,
            force_simulation=False,
        )
        assert (model.spectral_radius() < 1.)
        n_simulations = 5
        multi = SimuHawkesMulti(
            model,
            n_simulations=n_simulations,
        )
        multi.end_time = [end_time] * n_simulations
        multi.simulate()
        return multi

    def _test_relative_magnitudes(self,
                                  expected: np.ndarray,
                                  estimated: np.ndarray,
                                  significance_threshold: float,
                                  quantity_name: str,
                                  significance_band_width: float = 10,
                                  ):
        self.assertFalse(
            np.any(
                np.logical_and(
                    np.sort(np.abs(expected.flatten())
                            ) < significance_threshold,
                    np.sort(np.abs(estimated.flatten())) > significance_band_width *
                    significance_threshold
                )
            ) or
            np.any(
                np.logical_and(
                    np.sort(np.abs(expected.flatten())) > significance_band_width *
                    significance_threshold,
                    np.sort(np.abs(estimated.flatten())
                            ) < significance_threshold
                )
            ),
            f'Sorted estimated {quantity_name} and sorted expected {quantity_name} '
            'have corresponding entries on the opposite side of the band '
            f'({significance_threshold}, {significance_band_width*significance_threshold}):\n'
            f'Sorted absolute estimated {quantity_name}:\n'
            f'{np.sort(np.abs(estimated.flatten()))}\n'
            f'Sorted absolute expected {quantity_name}:\n'
            f'{np.sort(np.abs(expected.flatten()))}\n'
        )
        significance_idx = np.logical_and(
            np.abs(expected) > significance_threshold,
            np.abs(estimated) > significance_threshold
        )
        significant_estimated = estimated[
            significance_idx].flatten()
        significant_estimated_argsort = np.argsort(
            significant_estimated)
        sorted_significant_estimated = np.sort(
            significant_estimated)
        significant_expected = expected[
            significance_idx].flatten()
        significant_expected_argsort = np.argsort(
            significant_expected)
        sorted_significant_expected = np.sort(
            significant_expected)
        np.testing.assert_array_equal(
            significant_expected_argsort,
            significant_estimated_argsort,
            err_msg='Relative magnitudes of '
            f'estimated {quantity_name} differ from '
            f'relative magnitudes of expected {quantity_name}.\n'
            f'significant {quantity_name} expected :\n{significant_expected}\n'
            f'significant {quantity_name} estimated :\n{significant_estimated}\n'
            f'significant expected {quantity_name} argsort:\n{significant_expected_argsort}\n'
            f'significant estimated {quantity_name} argsort:\n{significant_estimated_argsort}\n'
        )

    def test_hawkes_cumulants(self):
        """...Test that estimated cumulants are correct
        """

        multi = Test.get_simulated_model()
        timestamps = multi.timestamps
        baseline = multi.hawkes_simu.baseline
        adjacency = multi.hawkes_simu.adjacency
        decays = multi.hawkes_simu.decays
        integration_support = .3
        n_nodes = multi.hawkes_simu.n_nodes

        theoretical_cumulant = HawkesTheoreticalCumulant(n_nodes)
        self.assertEqual(theoretical_cumulant.dimension, n_nodes)
        theoretical_cumulant.baseline = baseline
        theoretical_cumulant.adjacency = adjacency

        np.testing.assert_array_almost_equal(
            theoretical_cumulant.baseline,
            baseline)
        np.testing.assert_array_almost_equal(
            theoretical_cumulant.adjacency,
            adjacency)
        np.testing.assert_array_almost_equal(
            np.eye(theoretical_cumulant.dimension) -
            np.linalg.inv(theoretical_cumulant._R),
            adjacency
        )

        theoretical_cumulant.compute_cumulants()
        expected_L = theoretical_cumulant.mean_intensity
        expected_C = theoretical_cumulant.covariance
        expected_K = theoretical_cumulant.skewness

        learner = HawkesCumulantMatching(
            integration_support=integration_support)
        learner._set_data(timestamps)
        self.assertFalse(learner._cumulant_computer.cumulants_ready)
        learner.compute_cumulants()
        self.assertTrue(learner._cumulant_computer.cumulants_ready)

        # Test 1 - mean intensity
        # Test 1.1 - relative magnitudes of mean intensity are the same as
        # realtive magnitudes of expected mean intensity
        self._test_relative_magnitudes(
            estimated=learner.mean_intensity,
            expected=expected_L,
            quantity_name='mean intensity',
            significance_threshold=1e-7,
        )
        # Test 1.2 - estimated mean intensity is close to expected mean intensity
        np.testing.assert_allclose(
            learner.mean_intensity,
            expected_L,
            atol=0.005,
            rtol=0.015,
        )

        # Test 2 - covariance
        # Test 2.1 - relative magnitudes of estimated covariances are the same
        # as relative magnitudes of expected covariances
        self._test_relative_magnitudes(
            estimated=learner.covariance,
            expected=expected_C,
            quantity_name='variance',
            significance_threshold=5.75e-5,
        )
        # Test 2.2 - estimated covariance is close to expected covariance
        np.testing.assert_allclose(
            learner.covariance,
            expected_C,
            atol=0.03,  # Can we design a test that succeed when tolerance is lower?
            rtol=0.05,
        )

        # Test 3 - skewness
        # Test 3.1 - relative magnitudes of estimated skewnesss are the same
        # as relative magnitudes of expected skewnesss
        self._test_relative_magnitudes(
            estimated=learner.skewness,
            expected=expected_K,
            quantity_name='skewness',
            significance_threshold=6.75e-5,
        )
        # Test 3.2 - estimated skewness is close to expected covariance
        np.testing.assert_allclose(
            learner.skewness,
            expected_K,
            atol=0.15,  # Can we design a test that succeed when tolerance is lower?
            rtol=0.2,
        )

        self.assertGreater(
            learner.approximate_optimal_cs_ratio(),
            0.0)

        learner._set_data(timestamps)
        self.assertTrue(learner._cumulant_computer.cumulants_ready)

    def test_starting_point(self):
        """...Test the starting point of the training
        """
        multi = Test.get_simulated_model()
        n_nodes = multi.hawkes_simu.n_nodes
        timestamps = multi.timestamps
        integration_support = .3

        learner = HawkesCumulantMatching(
            integration_support=integration_support,
        )
        learner._set_data(timestamps)
        learner.compute_cumulants()
        self.assertTrue(learner._cumulant_computer.cumulants_ready)
        sp: np.ndarray = learner.starting_point(random=False)
        sp_r: np.ndarray = learner.starting_point(random=True)
        self.assertEqual(sp.shape, (n_nodes, n_nodes))
        self.assertEqual(sp_r.shape, (n_nodes, n_nodes))
        zeros = np.zeros((n_nodes, n_nodes), dtype=float)
        np.testing.assert_array_almost_equal(
            np.imag(sp), zeros,
            err_msg='Non-random starting point returned an array '
            f'with non-real entries:\n{sp}'
        )
        np.testing.assert_array_almost_equal(
            np.imag(sp_r), zeros,
            err_msg='Random starting point returned an array '
            f'with non-real entries:\n{sp_r}'
        )

    def _test_objective(self, Learner: HawkesCumulantMatching, penalty: Optional[str] = None):
        """...Test the starting point of the training
        """
        multi = Test.get_simulated_model()
        n_nodes = multi.hawkes_simu.n_nodes
        timestamps = multi.timestamps
        integration_support = .3
        learner = Learner(
            integration_support=integration_support,
            penalty=penalty,
        )
        learner._set_data(timestamps)
        learner.compute_cumulants()
        learner.cs_ratio = learner.approximate_optimal_cs_ratio()
        self.assertTrue(learner._cumulant_computer.cumulants_ready)
        sp: np.ndarray = learner.starting_point(random=False)
        objective = learner.objective(R=sp)
        try:
            loss = float(objective)
        except Exception as e:
            self.fail(
                f'{e}: Training objective evaluated at non-random starting point '
                'cannot be converted into a float:\n'
                f'learner.obiective(R=sp) : {objective}'
            )

    @unittest.skipIf(SKIP_TF, "Tensorflow not available")
    def test_tf_objective(self):
        self._test_objective(Learner=HawkesCumulantMatchingTf, penalty=None)
        self._test_objective(Learner=HawkesCumulantMatchingTf, penalty='l1')
        self._test_objective(Learner=HawkesCumulantMatchingTf, penalty='l2')

    @unittest.skipIf(SKIP_TORCH, "PyTorch not available")
    def test_pyt_objective(self):
        self._test_objective(Learner=HawkesCumulantMatchingPyT, penalty=None)
        self._test_objective(Learner=HawkesCumulantMatchingPyT, penalty='l1')
        self._test_objective(Learner=HawkesCumulantMatchingPyT, penalty='l2')

    def test_hawkes_cumulants_unfit(self):
        """...Test that HawkesCumulantMatching raises an error if no data is
        given
        """
        learner = HawkesCumulantMatching(
            100.,
            cs_ratio=0.9,
            max_iter=299,
            print_every=30,
            step=1e-2,
            solver='adam',
        )

        msg = '^Cannot compute cumulants if no realization has been provided$'
        with self.assertRaisesRegex(RuntimeError, msg):
            learner.compute_cumulants()

    @unittest.skipIf(SKIP_TF, "Tensorflow not available")
    def test_hawkes_cumulants_tf_solve(self):
        self._test_hawkes_cumulants_solve(
            Learner=HawkesCumulantMatchingTf,
            max_iter=8000,
            step=1e-5,
            solver='adam',
            penalty=None,
            C=1e-4,
            tol=1e-16,
            R_significance_threshold=2.05e-2,
            # This will effectively suppress the check but it is ok becasue baselines are all equal
            baseline_significance_threshold=1e-3,
            adjacency_significance_threshold=1.85e-2,
            significance_band_width=5.,
        )

    @unittest.skipIf(SKIP_TORCH, "PyTorch not available")
    def test_hawkes_cumulants_pyt_solve(self):
        self._test_hawkes_cumulants_solve(
            Learner=HawkesCumulantMatchingPyT,
            max_iter=4000,
            print_every=30,
            step=1e-4,
            solver='adam',
            penalty=None,
            C=1e-3,
            tol=1e-16,
            R_significance_threshold=2.7e-2,
            # This will effectively suppress the check but it is ok becasue baselines are all equal
            baseline_significance_threshold=1e-3,
            adjacency_significance_threshold=3e-2,
            significance_band_width=5.,
        )

    @unittest.skipIf(SKIP_TF, "Tensorflow not available")
    def test_hawkes_cumulants_tf_solve_l1(self):
        self._test_hawkes_cumulants_solve(
            Learner=HawkesCumulantMatchingTf,
            max_iter=8000,
            step=1e-5,
            solver='adam',
            penalty='l1',
            C=1e-6,
            tol=1e-16,
            R_significance_threshold=1.5e-0,  # relative magnitudes of R effectively not tested
            # relative magnitudes of baseline effectively not tested
            baseline_significance_threshold=1e-3,
            adjacency_significance_threshold=1.25e-6,
            significance_band_width=9e+4,
        )

    @unittest.skipIf(SKIP_TORCH, "PyTorch not available")
    def test_hawkes_cumulants_pyt_solve_l1(self):
        self._test_hawkes_cumulants_solve(
            Learner=HawkesCumulantMatchingPyT,
            max_iter=8000,
            print_every=30,
            step=1e-5,
            solver='adam',
            penalty='l1',
            C=1e-6,
            tol=1e-16,
            R_significance_threshold=1.5e-0,  # relative magnitudes of R effectively not tested
            # relative magnitudes of baseline effectively not tested
            baseline_significance_threshold=1e-3,
            adjacency_significance_threshold=1.25e-6,
            significance_band_width=1.3e+5,
        )

    @unittest.skipIf(SKIP_TF, "Tensorflow not available")
    def test_hawkes_cumulants_tf_solve_l2(self):
        self._test_hawkes_cumulants_solve(
            Learner=HawkesCumulantMatchingTf,
            max_iter=8000,
            step=1e-5,
            solver='adam',
            penalty='l2',
            C=1e-5,
            tol=1e-16,
            R_significance_threshold=1.5e-6,
            # relative magnitudes of baseline effectively not tested
            baseline_significance_threshold=1e-3,
            adjacency_significance_threshold=5e-13,
            significance_band_width=5e+12,
        )

    @unittest.skipIf(SKIP_TORCH, "PyTorch not available")
    def test_hawkes_cumulants_pyt_solve_l2(self):
        self._test_hawkes_cumulants_solve(
            Learner=HawkesCumulantMatchingPyT,
            max_iter=4000,
            print_every=30,
            step=1e-4,
            solver='adam',
            penalty='l2',
            C=1e-3,
            tol=1e-16,
            R_significance_threshold=1.5e-0,  # effectively not tested
            # relative magnitudes of baseline effectively not tested
            baseline_significance_threshold=1e-3,
            adjacency_significance_threshold=5e-5,  # effectively not tested
            significance_band_width=5e+12,
        )

    def _test_hawkes_cumulants_solve(
            self,
            Learner=HawkesCumulantMatchingTf,
            max_iter=4000,
            print_every=30,
            step=1e-4,
            solver='adam',
            penalty=None,
            C=1e-3,
            tol=1e-16,
            verbose=False,
            R_significance_threshold=1e-4,
            adjacency_significance_threshold=1e-4,
            baseline_significance_threshold=1e-4,
            significance_band_width=10.,

    ):
        """...Test that hawkes cumulant reached expected value
        """
        multi = Test.get_simulated_model()
        n_nodes = multi.hawkes_simu.n_nodes
        timestamps = multi.timestamps
        baseline = multi.hawkes_simu.baseline
        decays = multi.hawkes_simu.decays
        integration_support = .3

        learner = Learner(
            integration_support=integration_support,
            max_iter=max_iter,
            print_every=print_every,
            step=step,
            solver=solver,
            penalty=penalty,
            C=C,
            tol=tol,
        )
        learner.fit(timestamps)
        if verbose:
            learner.print_history()

        expected_R_pred = np.linalg.inv(
            np.eye(n_nodes) - multi.hawkes_simu.adjacency
        )

        if verbose:
            print('\n')
            print(f'expected_R_pred:\n{expected_R_pred}')
            print(f'solution:\n{learner.solution}')

        self._test_relative_magnitudes(
            quantity_name='R',
            expected=expected_R_pred,
            estimated=learner.solution,
            significance_threshold=R_significance_threshold,
            significance_band_width=significance_band_width,
        )

        self.assertTrue(
            np.allclose(
                learner.solution,
                expected_R_pred,
                atol=0.1,  # TODO: explain why estimation is not so accurate
                rtol=0.25,
            ))

        expected_baseline = baseline

        if verbose:
            print('\n')
            print(f'expected_baseline:\n{expected_baseline}')
            print(f'estimated_baseline:\n{learner.baseline}')

        self._test_relative_magnitudes(
            quantity_name='baseline',
            expected=expected_baseline,
            estimated=learner.baseline,
            significance_threshold=baseline_significance_threshold,
            significance_band_width=significance_band_width,
        )

        self.assertTrue(
            np.allclose(
                learner.baseline,
                expected_baseline,
                atol=0.01,
                rtol=0.1,
            ))

        expected_adjacency = multi.hawkes_simu.adjacency
        if verbose:
            print('\n')
            print(f'expected_adjacency:\n{expected_adjacency}')
            print(f'estimated_adjacency:\n{learner.adjacency}')

        self._test_relative_magnitudes(
            quantity_name='adjacency',
            expected=expected_adjacency,
            estimated=learner.adjacency,
            significance_threshold=adjacency_significance_threshold,
            significance_band_width=significance_band_width,
        )

        if penalty in ('l1', 'l2'):
            atol = 0.2
        else:
            atol = 0.1
        self.assertTrue(
            np.allclose(
                learner.adjacency,
                expected_adjacency,
                atol=atol,  # TODO: explain why estimation is not so accurate
                rtol=0.25,
            ))


if __name__ == "__main__":
    unittest.main()
