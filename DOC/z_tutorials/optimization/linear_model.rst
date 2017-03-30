Generalized linear models
=========================

This example illustrates the opimization of three linear models:
    * Linear regression (`tick.optim.model.ModelLinReg`)
    * Logistic regression (`tick.optim.model.ModelLogReg`)
    * Poisson regression (`tick.optim.model.ModelPoisReg`)
with five different solvers:
    * LBFGS (`tick.optim.solver.BFGS`)
    * SVRG (`tick.optim.solver.SVRG`)
    * SDCA (`tick.optim.solver.SDCA`)
    * GD (`tick.optim.solver.GD`)
    * AGD (`tick.optim.solver.AGD`)

.. plot:: z_tutorials/optimization/code_samples/linear_model.py
    :include-source:
