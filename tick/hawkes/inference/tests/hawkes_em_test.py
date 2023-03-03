# License: BSD 3 clause

import unittest

import numpy as np

from tick.hawkes.inference import HawkesEM
from tick.hawkes.model.tests.model_hawkes_test_utils import (
    hawkes_intensities, hawkes_log_likelihood)
from tick.hawkes import (SimuHawkes, HawkesKernelTimeFunc,
                         HawkesKernelExp, SimuHawkesExpKernels)
from tick.base import TimeFunction


def simulate_hawkes_exp_kern(
        decays=[[1., 1.5], [0.1, 0.5]],
        baseline=[0.12, 0.07],
        adjacency=[[.1, .4], [.2, 0.5]],
        end_time=3000,
        max_jumps=1000,
        verbose=False,
        force_simulation=False,
        seed=None,
):
    model = SimuHawkesExpKernels(
        adjacency=adjacency,
        decays=decays,
        baseline=baseline,
        end_time=end_time,
        max_jumps=max_jumps,
        verbose=verbose,
        force_simulation=force_simulation,
        seed=seed,
    )
    model.track_intensity(intensity_track_step=0.1)
    model.simulate()
    return model


def compute_approx_support_of_exp_kernel(a, b, eps):
    return np.maximum(0., np.squeeze(np.max(- np.log(eps / (a*b)) / b)))


def simulate_hawkes_nonparam_kern(
        end_time=30000,
        seed=None
):
    t_values1 = np.array([0, 1, 1.5, 2., 3.5], dtype=float)
    y_values1 = np.array([0, 0.2, 0, 0.1, 0.], dtype=float)
    tf1 = TimeFunction([t_values1, y_values1],
                       inter_mode=TimeFunction.InterConstRight, dt=0.1)
    kernel1 = HawkesKernelTimeFunc(tf1)

    t_values2 = np.linspace(0, 4, 20)
    y_values2 = np.maximum(0., np.sin(t_values2) / 4)
    tf2 = TimeFunction([t_values2, y_values2])
    kernel2 = HawkesKernelTimeFunc(tf2)

    baseline = np.array([0.1, 0.3])

    hawkes = SimuHawkes(
        baseline=baseline,
        end_time=end_time,
        verbose=False,
        seed=seed,
    )

    hawkes.set_kernel(0, 0, kernel1)
    hawkes.set_kernel(0, 1, HawkesKernelExp(.5, .7))
    hawkes.set_kernel(1, 1, kernel2)

    hawkes.simulate()
    return hawkes


def discretization_of_exp_kernel(
        n_nodes, a, b,  kernel_support, kernel_size):
    assert (n_nodes, n_nodes) == a.shape
    assert a.shape == b.shape
    assert kernel_size > 0
    assert kernel_support > .0
    abscissa = np.linspace(0, kernel_support, kernel_size)
    abscissa_ = np.expand_dims(abscissa, axis=0)
    abscissa_ = np.repeat(abscissa_, repeats=n_nodes, axis=0)
    abscissa__ = np.expand_dims(abscissa_, axis=0)
    abscissa__ = np.repeat(abscissa__, repeats=n_nodes, axis=0)
    assert abscissa_.shape == (n_nodes, kernel_size)
    assert abscissa__.shape == (n_nodes, n_nodes, kernel_size)
    a_ = np.repeat(np.expand_dims(a, axis=-1),
                   repeats=kernel_size, axis=2)
    b_ = np.repeat(np.expand_dims(b, axis=-1),
                   repeats=kernel_size, axis=2)
    assert a_.shape == (n_nodes, n_nodes, kernel_size)
    assert a_.shape == b_.shape
    assert a_.shape == abscissa__.shape
    res_ = a_ * b_ * np.exp(-b_ * abscissa__)
    assert res_.shape == (n_nodes, n_nodes, kernel_size)
    return res_


class Test(unittest.TestCase):
    def setUp(self):
        np.random.seed(123269)
        self.n_nodes = 3
        self.n_realizations = 2

        self.events = [[
            np.cumsum(np.random.rand(4 + i)) for i in range(self.n_nodes)
        ] for _ in range(self.n_realizations)]

    def test_hawkes_em_attributes(self):
        """...Test attributes of HawkesEM are correctly inherited
        """
        em = HawkesEM(kernel_support=10)
        em.fit(self.events)
        self.assertEqual(em.n_nodes, self.n_nodes)
        self.assertEqual(em.n_realizations, self.n_realizations)

    def test_hawkes_em_fit_1(self):  # hard-coded
        """...Test fit method of HawkesEM
        """
        kernel_support = 3
        kernel_size = 3
        baseline = np.zeros(self.n_nodes) + .2
        kernel = np.zeros((self.n_nodes, self.n_nodes, kernel_size)) + .4

        em = HawkesEM(kernel_support=kernel_support, kernel_size=kernel_size,
                      n_threads=2, max_iter=11, verbose=False)
        em.fit(self.events, baseline_start=baseline, kernel_start=kernel)

        np.testing.assert_array_almost_equal(
            em.baseline, [1.2264, 0.2164, 1.6782], decimal=4)

        expected_kernel = [[[2.4569e-02, 2.5128e-06,
                             0.0000e+00], [1.8072e-02, 5.4332e-11, 0.0000e+00],
                            [2.7286e-03, 4.0941e-08, 3.5705e-15]],
                           [[8.0077e-01, 2.2624e-02,
                             6.7577e-10], [2.7503e-02, 3.1840e-05, 0.0000e+00],
                            [1.4984e-01, 7.8428e-06, 2.8206e-12]],
                           [[1.2163e-01, 1.0997e-02,
                             5.4724e-05], [4.7348e-02, 6.6093e-03, 5.5433e-12],
                            [1.0662e-03, 5.3920e-05, 1.4930e-08]]]

        np.testing.assert_array_almost_equal(em.kernel, expected_kernel,
                                             decimal=4)

        em2 = HawkesEM(
            kernel_discretization=np.array([0., 1., 2., 3.]), n_threads=1,
            max_iter=11, verbose=False)
        em2.fit(self.events, baseline_start=baseline, kernel_start=kernel)
        np.testing.assert_array_almost_equal(em2.kernel, expected_kernel,
                                             decimal=4)

        np.testing.assert_array_almost_equal(
            em.get_kernel_values(1, 0, np.linspace(0, 3, 5)),
            [0.0000e+00, 8.0077e-01, 2.2624e-02, 6.7577e-10, 0.0000e+00],
            decimal=4)

        np.testing.assert_array_almost_equal(
            em.get_kernel_norms(),
            [[0.0246, 0.0181, 0.0027], [0.8234, 0.0275, 0.1499],
             [0.1327, 0.054, 0.0011]], decimal=3)

        np.testing.assert_array_equal(
            em.get_kernel_supports(),
            np.ones((self.n_nodes, self.n_nodes)) * 3)

    # With simulated data from parametric (exponential) model
    def test_hawkes_em_fit_2(self):
        """
        Test estimation on simulated data from a parametric 
        Hawkes model with exponential kernels
        """
        seed = 12345
        n_nodes = 2
        n_realizations = 1
        decays = np.array([[1., 1.5], [0.1, 0.5]])
        baseline = np.array([0.12, 0.07])
        adjacency = np.array([[.1, .4], [.2, 0.03]])
        end_time = 30000
        simu_model = simulate_hawkes_exp_kern(
            decays=decays,
            baseline=baseline,
            adjacency=adjacency,
            end_time=end_time,
            max_jumps=200000,
            verbose=True,
            force_simulation=False,
            seed=seed,
        )

        kernel_support = compute_approx_support_of_exp_kernel(
            adjacency, decays, 1e-4)
        # print(f'kernel_suppport = {kernel_support}')
        kernel_size = 20

        events = simu_model.timestamps
        baseline_start = np.array([.05 * np.mean(np.diff(ts))
                                   for ts in simu_model.timestamps])
        # print(f'baseline_start = {baseline_start}')
        kernel_start = np.zeros((n_nodes, n_nodes, kernel_size))
        kernel_start[:, :, :kernel_size-1] = .01 * np.cumsum(
            np.random.uniform(size=(n_nodes, n_nodes, kernel_size-1)), axis=2)[:, :, ::-1]
        # print(f'kernel_start = {kernel_start}')
        em = HawkesEM(kernel_support=kernel_support,
                      kernel_size=kernel_size,
                      tol=1e-9,
                      print_every=50,
                      record_every=10,
                      max_iter=1200,
                      verbose=True)
        em.fit(events, baseline_start=baseline_start,
               kernel_start=kernel_start)

        expected_kernel = discretization_of_exp_kernel(
            n_nodes,
            adjacency,
            decays,
            kernel_support,
            kernel_size,
        )
        expected_baseline = baseline

        # Test 1.1: shape of baseline
        self.assertEqual(
            expected_baseline.shape,
            em.baseline.shape,
            'Expected baseline and estimated baseline do not have the same shape\n'
            'expected_baseline.shape: {expected_baseline.shape}\n'
            'em.baseline.shape: {em.baseline.shape}\n'
        )

        # Test 1.2: relative magnitudes of baseline
        np.testing.assert_array_equal(
            np.argsort(em.baseline),
            np.argsort(expected_baseline),
            err_msg='Relative magnitudes are not consistent'
            'between expected baseline and estimated baseline',
        )

        # Test 1.3: approximate equality of baseline
        np.testing.assert_array_almost_equal(
            em.baseline,
            expected_baseline,
            decimal=1,
        )

        # Test 2.1: shape of kernel
        self.assertEqual(
            expected_kernel.shape,
            em.kernel.shape,
            'Expected kernel and estimated kernel do not have the same shape\n'
            'expected_kernel.shape: {expected_kernel.shape}\n'
            'em.kernel.shape: {em.kernel.shape}\n'
        )

        # Test 2.2: estimated kernel must be non-negative
        self.assertTrue(
            np.all(em.kernel >= 0),
            f'Estimated kernel takes negative values'
        )

        # Test 2.3: estimated kernel must be non-increasing
        significance_threshold = 7e-3  # Can we do better?
        estimated_kernel_increments = np.diff(em.kernel, append=0., axis=2)
        estimated_kernel_significant_increments = estimated_kernel_increments[
            np.abs(estimated_kernel_increments) > significance_threshold
        ]
        assert estimated_kernel_increments.shape == (
            n_nodes, n_nodes, kernel_size)

        self.assertTrue(
            np.all(
                estimated_kernel_significant_increments <= 0.
            ),
            'Estimated kernel are not non-increasing functions. '
            'This is not compatible with the simulation '
            'being performed with exponential kernels.\n'
            f'estimated_kernel_increments:\n{estimated_kernel_increments}\n'
            f'estimated_kernel_significant_increments:\n{estimated_kernel_significant_increments}\n'
            f'estimated_kernel_significant_increments bigger than 0:\n{estimated_kernel_significant_increments[estimated_kernel_significant_increments>.0]}\n'
        )

        # Test 2.4: relative magnitudes of kernel at zero
        expected_kernel_at_zero = expected_kernel[:, :, 0].flatten()
        estimated_kernel_at_zero = np.array(em.kernel, copy=True)[
            :, :, 0].flatten()
        np.testing.assert_array_equal(
            np.argsort(
                estimated_kernel_at_zero[estimated_kernel_at_zero > .0]),
            np.argsort(expected_kernel_at_zero[expected_kernel_at_zero > .0]),
            err_msg='Relative magnitudes are not consistent '
            'between expected kernel and estimated kernel\n'
            f'expected_kernel_at_zero:\n{expected_kernel_at_zero}\n'
            f'estimated_kernel_at_zero:\n{estimated_kernel_at_zero}\n'
        )

        # Test 2.5: approximate equality of kernel at zero
        np.testing.assert_array_almost_equal(
            expected_kernel_at_zero,
            estimated_kernel_at_zero,
            decimal=0,  # Can we do better?
            err_msg='estimated kernel at zero deviates '
            'from expected  kernel at zero.\n'
            f'expected_kernel_at_zero:\n{expected_kernel_at_zero}\n'
            f'estimated_kernel_at_zero:\n{estimated_kernel_at_zero}\n'
        )

    def test_hawkes_em_fit_3(self):
        """Test estimation on simulated data from a non-parametric Hawkes model

        The test compares the time-integral of Hawkes kernels: 
        expected (i.e. exact, coming from the model used to simulate the data)
        and estimated (i.e. coming from the estimated kernels).
        """
        simu_model = simulate_hawkes_nonparam_kern(
            end_time=30000,
            seed=2334,
        )

        em = HawkesEM(
            kernel_support=4.,
            kernel_size=32,
            n_threads=8,
            verbose=True,
            max_iter=500,
            tol=1e-6)
        em.fit(simu_model.timestamps)

        evaluation_points = np.linspace(0, 4., num=10)
        for i in range(2):
            for j in range(2):
                estimated_primitive_kernel_values = em._compute_primitive_kernel_values(
                    i, j, evaluation_points)
                expected_primitive_kernel_values = simu_model.kernels[i, j].get_primitive_values(
                    evaluation_points)
                self.assertTrue(
                    np.allclose(
                        expected_primitive_kernel_values,
                        estimated_primitive_kernel_values,
                        atol=2e-2,
                        rtol=3.5-1,
                    ),
                    f'Kernel[{i}, {j}]: Estimation error\n'
                    f'Estimated values:\n{estimated_primitive_kernel_values}\n'
                    f'Expected values:\n{expected_primitive_kernel_values}\n'
                )

    def test_hawkes_em_score(self):
        """...Test score (ie. likelihood) function of Hawkes EM
        """

        def approximate_likelihood(em, events, end_times, precision=2):
            n_total_jumps = sum(map(len, events))
            kernels_func = [[
                lambda t, i=i, j=j: em.get_kernel_values(
                    i, j, np.array([t]))[0]
                for j in range(n_nodes)
            ] for i in range(n_nodes)]
            intensities = hawkes_intensities(events, em.baseline, kernels_func)
            return hawkes_log_likelihood(intensities, events, end_times,
                                         precision=precision) / n_total_jumps

        # We use only 2 nodes otherwise integral approximation might be very
        # slow
        n_nodes = 2
        kernel_support = 1.
        kernel_size = 3
        baseline = np.random.rand(n_nodes) + .2
        kernel = np.random.rand(n_nodes, n_nodes, kernel_size) + .4

        train_events = \
            [np.cumsum(np.random.rand(2 + i)) for i in range(n_nodes)]

        test_events = \
            [2 + np.cumsum(np.random.rand(2 + i)) for i in range(n_nodes)]

        # Test for 2 kind of discretization
        train_kwargs = [{
            'kernel_support': 1,
            'kernel_size': 3
        }, {
            'kernel_discretization': np.array([0., 1., 1.5, 3.])
        }]

        # Test with and without fitting
        fits = [True, False]

        for kwargs, fit in zip(train_kwargs, fits):
            em = HawkesEM(**kwargs)
            end_times = max(map(max, train_events)) + 0.2 * kernel_support

            msg = '^You must either call `fit` before `score` or provide events'
            with self.assertRaisesRegex(ValueError, msg):
                em.score()

            if fit:
                em.fit(train_events, end_times=end_times,
                       baseline_start=baseline, kernel_start=kernel)
            else:
                em.baseline = baseline
                em.kernel = kernel

            # Score on em train data
            if fit:
                em_train_score = em.score()
            else:
                em_train_score = em.score(train_events, end_times=end_times)
            self.assertAlmostEqual(
                em_train_score,
                approximate_likelihood(em, train_events, end_times, 2),
                delta=1e-1, msg='Failed on train for {}'.format(kwargs))

            # Score on test data
            em_test_score = em.score(events=test_events)
            test_end_times = max(map(max, test_events))
            self.assertAlmostEqual(
                em_test_score,
                approximate_likelihood(em, test_events, test_end_times, 4),
                delta=1e-3, msg='Failed on test for {}'.format(kwargs))

    def test_hawkes_em_kernel_shape(self):
        kernel_support = 4
        kernel_size = 10
        em = HawkesEM(kernel_support=kernel_support, kernel_size=kernel_size,
                      n_threads=2, max_iter=11, verbose=False)
        em.fit(self.events)
        reshaped_kernel = em._flat_kernels.reshape((
            em.n_nodes, em.n_nodes, em.kernel_size))
        self.assertTrue(
            np.allclose(em.kernel,                reshaped_kernel
                        ),
            "Reshaping of kernel is inconsistent:"
            f"kernel: {em.kernel}\n"
            f"reshaped kernel: {reshaped_kernel}\n"
        )

    def test_hawkes_em_kernel_support(self):
        """...Test that Hawkes em kernel support parameter is correctly
        synchronized
        """
        kernel_support_1 = 4.4
        learner = HawkesEM(kernel_support_1)
        self.assertEqual(learner.kernel_support, kernel_support_1)
        self.assertEqual(learner._learner.get_kernel_support(),
                         kernel_support_1)
        expected_kernel_discretization = [
            0.0, 0.44, 0.88, 1.32, 1.76, 2.2, 2.64, 3.08, 3.52, 3.96, 4.4
        ]
        np.testing.assert_array_almost_equal(learner.kernel_discretization,
                                             expected_kernel_discretization)

        kernel_support_2 = 6.2
        learner.kernel_support = kernel_support_2
        self.assertEqual(learner.kernel_support, kernel_support_2)
        self.assertEqual(learner._learner.get_kernel_support(),
                         kernel_support_2)

        expected_kernel_discretization = [
            0.0, 0.62, 1.24, 1.86, 2.48, 3.1, 3.72, 4.34, 4.96, 5.58, 6.2
        ]
        np.testing.assert_array_almost_equal(learner.kernel_discretization,
                                             expected_kernel_discretization)

    def test_hawkes_em_kernel_size(self):
        """...Test that Hawkes em kernel size parameter is correctly
        synchronized
        """
        kernel_size_1 = 4
        learner = HawkesEM(4., kernel_size=kernel_size_1)
        self.assertEqual(learner.kernel_size, kernel_size_1)
        self.assertEqual(learner._learner.get_kernel_size(), kernel_size_1)
        expected_kernel_discretization = [0., 1., 2., 3., 4.]
        np.testing.assert_array_almost_equal(learner.kernel_discretization,
                                             expected_kernel_discretization)

        kernel_size_2 = 5
        learner.kernel_size = kernel_size_2
        self.assertEqual(learner.kernel_size, kernel_size_2)
        self.assertEqual(learner._learner.get_kernel_size(), kernel_size_2)
        expected_kernel_discretization = [0.0, 0.8, 1.6, 2.4, 3.2, 4]
        np.testing.assert_array_almost_equal(learner.kernel_discretization,
                                             expected_kernel_discretization)

    def test_hawkes_em_kernel_dt(self):
        """...Test that Hawkes em kernel dt parameter is correctly
        synchronized
        """
        kernel_support = 4
        kernel_size = 10
        learner = HawkesEM(kernel_support, kernel_size=kernel_size)
        self.assertEqual(learner.kernel_dt, 0.4)
        self.assertEqual(learner._learner.get_kernel_fixed_dt(), 0.4)

        kernel_dt_1 = 0.2
        learner.kernel_dt = kernel_dt_1
        self.assertEqual(learner.kernel_dt, kernel_dt_1)
        self.assertEqual(learner._learner.get_kernel_fixed_dt(), kernel_dt_1)
        self.assertEqual(learner.kernel_size, 20)
        expected_kernel_discretization = [
            0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6,
            2.8, 3., 3.2, 3.4, 3.6, 3.8, 4.
        ]
        np.testing.assert_array_almost_equal(learner.kernel_discretization,
                                             expected_kernel_discretization)

        kernel_dt_1 = 0.199
        learner.kernel_dt = kernel_dt_1
        self.assertEqual(learner.kernel_dt, 0.19047619047619047)
        self.assertEqual(learner._learner.get_kernel_fixed_dt(),
                         0.19047619047619047)
        self.assertEqual(learner.kernel_size, 21)

    def test_hawkes_em_get_kernel_values(self):
        kernel_support = 4
        kernel_size = 10
        em = HawkesEM(kernel_support=kernel_support, kernel_size=kernel_size,
                      n_threads=2, max_iter=11, verbose=False)
        em.fit(self.events)

        # Test 0
        self.assertEqual(em.kernel.shape, (self.n_nodes, self.n_nodes, kernel_size),
                         'Estimated kernel has wrong shape'
                         f'Expected shape: {(self.n_nodes, self.n_nodes, kernel_size)}\n'
                         f'Shape of estimated kernel: {em.kernel.shape}\n'
                         )
        # Test 1
        self.assertTrue(np.all(em.kernel >= 0.),
                        "Error: kernel cannot take negative values!")

        # Test 2
        # The method `get_kernel_values` when evaluated at the
        # discrtetization point of the kernel must yield the values stored in `kernel`
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                vals = em.get_kernel_values(
                    i, j, em.kernel_discretization[1:])
                self.assertTrue(np.allclose(vals, em.kernel[i, j, :]))

    def test_hawkes_em_kernel_primitives(self):
        kernel_support = 4
        kernel_size = 10
        em = HawkesEM(kernel_support=kernel_support, kernel_size=kernel_size,
                      n_threads=2, max_iter=11, verbose=False)
        em.fit(self.events)
        # Pre-test 1
        self.assertTrue(np.all(em.kernel >= 0.),
                        "Error: kernel cannot take negative values!")
        # Pre-test 2
        primitives = em._get_kernel_primitives()
        self.assertEqual(primitives.shape,
                         (self.n_nodes, self.n_nodes, kernel_size),
                         "Erorr : Shape of primitives does not match expected shape"
                         )

        # Test 0
        # Primitives must be non-decreasing  functions, since kernels are non-negative
        self.assertTrue(
            np.all(np.diff(primitives, axis=2) >= 0.),
            "Error: primitives is not non-decreasing"
        )

        # Test 1
        # Since kernels are positive, their primitive evaluated at the
        # rightmost point of their support is expected to be equal to their norm
        norms = em.get_kernel_norms()
        self.assertTrue(
            np.allclose(norms, primitives[:, :,  -1]),
            "The values of the kernel primitives at the end of the kernel supports,"
            "must agree with the kernel norms."
        )

        # Test 2
        # The method `_compute_primitive_kernel_values` when evaluated at the
        # discrtetization point of the kernel must yield the values stored in `primitive`
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                vals = em._compute_primitive_kernel_values(
                    i, j, em.kernel_discretization[1:])
                self.assertTrue(np.allclose(vals, primitives[i, j, :]))

        # Test 3
        # We test that the values of the primitives computed via the class method agree
        # with the primitives of the kernels computed using numpy arrays
        def _compute_primitive_with_numpy(kernel, kernel_discretization):
            n = kernel.shape[0]
            m = kernel.shape[1]
            l = kernel.shape[2]
            assert len(kernel_discretization) == l + 1
            steps = np.diff(kernel_discretization)
            steps = np.repeat(steps.reshape((1, l)), repeats=m, axis=0)
            steps = np.repeat(steps.reshape((1, m, l)), repeats=n, axis=0)
            assert steps.shape == (n, m, l)
            assert steps.shape == kernel.shape
            primitives = np.cumsum(kernel*steps, axis=2)
            assert primitives.shape == (n, m, l)
            return primitives
        self.assertTrue(np.allclose(primitives, _compute_primitive_with_numpy(
            em.kernel, em.kernel_discretization)))

    def test_time_changed_interarrival_times_exp_kern(self):
        seed = 12345
        n_nodes = 2
        n_realizations = 1
        decays = np.array([[1., 1.5], [0.1, 0.5]])
        baseline = np.array([0.12, 0.07])
        adjacency = np.array([[.1, .4], [.2, 0.03]])
        end_time = 30000
        simu_model = simulate_hawkes_exp_kern(
            decays=decays,
            baseline=baseline,
            adjacency=adjacency,
            end_time=end_time,
            max_jumps=200000,
            verbose=True,
            force_simulation=False,
            seed=seed,
        )
        simu_model.store_compensator_values()
        simu_time_changed_interarrival_times = [
            [np.diff(c) for c in simu_model.tracked_compensator]
        ]

        kernel_support = compute_approx_support_of_exp_kernel(
            adjacency, decays, 1e-4)
        # print(f'kernel_suppport = {kernel_support}')
        kernel_size = 20

        events = [simu_model.timestamps]

        em = HawkesEM(kernel_support=kernel_support,
                      kernel_size=kernel_size,
                      tol=1e-9,
                      print_every=50,
                      record_every=10,
                      max_iter=1200,
                      verbose=True)

        # Test Part 1 - Exact parameters
        discretized_kernel = discretization_of_exp_kernel(
            n_nodes,
            adjacency,
            decays,
            kernel_support,
            kernel_size,
        )
        tcit = em.time_changed_interarrival_times(
            events=events,
            end_times=end_time,
            baseline=baseline,
            kernel=discretized_kernel,
        )
        for r in range(n_realizations):
            for e in range(n_nodes):
                self.assertTrue(
                    np.all(tcit[r][e] > .0),
                    'Exact parameters: Assertion error: '
                    'Inter-arrival times of '
                    f'realization {r} and component {e}'
                    ' are not all positive'
                )
        for r in range(n_realizations):
            for e in range(n_nodes):
                self.assertAlmostEqual(
                    np.quantile(
                        simu_time_changed_interarrival_times[r][e], .55),
                    1.,
                    None,
                    msg=f'Realization {r}, component {e}:\n'
                    'Time-changed inter-arrival times '
                    'as computed from simulation '
                    'are not distributed around 1.',
                    delta=.35,  # Can we do better?
                )
                self.assertAlmostEqual(
                    np.quantile(tcit[r][e], .5),
                    1.,
                    None,
                    msg='Exact parameters: Assertion error: '
                    f'Realization {r}, component {e}:\n'
                    'Time-changed inter-arrival times '
                    'as computed from estimation '
                    'are not distributed around 1.',
                    delta=.35,  # Can we do better?
                )

        # Test - Part 2 - Estimated parameters
        baseline_start = np.array([.05 * np.mean(np.diff(ts))
                                   for ts in simu_model.timestamps])
        kernel_start = np.zeros((n_nodes, n_nodes, kernel_size))
        kernel_start[:, :, :kernel_size-1] = .01 * np.cumsum(
            np.random.uniform(size=(n_nodes, n_nodes, kernel_size-1)), axis=2)[:, :, ::-1]
        em.fit(events, baseline_start=baseline_start,
               kernel_start=kernel_start)
        tcit = em.time_changed_interarrival_times()
        for r in range(n_realizations):
            for e in range(n_nodes):
                self.assertTrue(
                    np.all(tcit[r][e] > .0),
                    'Estimated parameters: Assertion error: '
                    'Inter-arrival times of '
                    f'realization {r} and component {e}'
                    ' are not all positive'
                )
        for r in range(n_realizations):
            for e in range(n_nodes):
                self.assertAlmostEqual(
                    np.quantile(tcit[r][e], .5),
                    1.,
                    None,
                    msg='Estimated parameters: Assertion error: '
                    f'Realization {r}, component {e}:\n'
                    'Time-changed inter-arrival times '
                    'as computed from estimation '
                    'are not distributed around 1.',
                    delta=.35,  # Can we do better?
                )


if __name__ == "__main__":
    unittest.main()
