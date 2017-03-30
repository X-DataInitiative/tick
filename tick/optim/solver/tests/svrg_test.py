import unittest

from tick.optim.solver import SVRG
from tick.optim.solver.tests.solver import TestSolver
from tick.optim.solver.build.solver import SVRG as _SVRG


class Test(TestSolver):
    def test_solver_svrg(self):
        """...Check SVRG solver for a Logistic Regression with Ridge
        penalization
        """
        solver = SVRG(step=1e-3, max_iter=100, verbose=False, tol=0)
        self.check_solver(solver, fit_intercept=True, model="logreg",
                          decimal=1)

    def test_svrg_sparse_and_dense_consistency(self):
        """...Test SVRG can run all glm models and is consistent with sparsity
        """

        def create_solver():
            return SVRG(max_iter=1, verbose=False, step=1e-5,
                        seed=TestSolver.sto_seed)

        self._test_solver_sparse_and_dense_consistency(create_solver)

    def test_variance_reduction_setting(self):
        """...Test SVRG variance_reduction parameter is correctly set
        """
        svrg = SVRG()
        self.assertEqual(svrg.variance_reduction, 'last')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         _SVRG.VarianceReductionMethod_Last)

        svrg = SVRG(variance_reduction='rand')
        self.assertEqual(svrg.variance_reduction, 'rand')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         _SVRG.VarianceReductionMethod_Random)

        svrg.variance_reduction = 'avg'
        self.assertEqual(svrg.variance_reduction, 'avg')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         _SVRG.VarianceReductionMethod_Average)

        svrg.variance_reduction = 'rand'
        self.assertEqual(svrg.variance_reduction, 'rand')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         _SVRG.VarianceReductionMethod_Random)

        svrg.variance_reduction = 'last'
        self.assertEqual(svrg.variance_reduction, 'last')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         _SVRG.VarianceReductionMethod_Last)

        with self.assertRaises(ValueError):
            svrg.variance_reduction = 'wrong_name'

if __name__ == '__main__':
    unittest.main()
