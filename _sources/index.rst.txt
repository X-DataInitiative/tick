.. title:: Tick

.. raw:: html

  <div
    class="jumbotron"
    style="padding-top: 10px; padding-bottom: 30px; position: relative"
  >
     <div class="container">
        <h1 style="font-size:40px">tick</h1>
        <p style="font-size:18px">
           tick a machine learning library for Python 3.
           The focus is on statistical learning for time dependent systems,
           such as point processes. Tick features also tools for generalized
           linear models, and a generic optimization tools, including solvers
           and proximal operators for penalization of model weights.
           It comes also with a bunch of tools for the simulation of datasets.
        </p>
        <a class="btn btn-primary btn-lg" href="auto_examples/index.html"
        role="button">
           Show me examples !
        </a>
     </div>
     <a href="https://github.com/X-DataInitiative/tick">
       <img style="position: absolute; top: 0; right: 0"
         src="_static/images/fork_me_on_github.png">
     </a>
  </div>

  <!-- tick.hawkes and tick.linear_model -->

  <div class="row">
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/hawkes.html">
           <h2>tick.hawkes</h2>
        </a>
        <p>
           Inference and simulation of Hawkes processes, with both
           parametric and non-parametric estimation
           techniques and flexible tools for simulation.
        </p>
     </div>
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/linear_model.html">
           <h2>tick.linear_model</h2>
        </a>
        <p>
            Inference and simulation of linear models, including among others linear,
            logistic and Poisson regression, with a large set of penalization
            techniques and solvers.
        </p>
     </div>
 </div>

 <!-- tick.robust and tick.survival -->

  <div class="row">
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/robust.html">
           <h2>tick.robust</h2>
        </a>
        <p>
            Tools for robust inference. It features tools for outliers detection
            and models such as Huber regression, among others robust losses.
        </p>
     </div>
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/survival.html">
           <h2>tick.survival</h2>
        </a>
        <p>
            Inference and simulation for survival analysis, including
            Cox regression with several penalizations.
        </p>
     </div>
 </div>

  <!-- tick.prox and tick.solver -->

  <div class="row">
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/prox.html">
           <h2>tick.prox</h2>
        </a>
        <p>
           Proximal operators for penalization of models weights. Such an
           operator can be used with (almost) any model and any solver.
        </p>
     </div>
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/solver.html">
           <h2>tick.solver</h2>
        </a>
        <p>
           A module that provides a bunch of state-of-the-art optimization
           algorithms, including both batch and stochastic solvers
        </p>
     </div>
 </div>

  <!-- tick.simulation and tick.plot -->

  <div class="row">
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/simulation.html">
           <h2>tick.simulation</h2>
        </a>
        <p>Basic tools for simulation, such as simulation of model weights and
            feature matrices.
        </p>
     </div>
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/plot.html">
           <h2>tick.plot</h2>
        </a>
        <p>
            Some plotting utilities used in tick, such as plots for point
            processes and solver convergence.
        </p>
     </div>
 </div>

  <!-- tick.dataset and tick.preprocessing -->

  <div class="row">
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/dataset.html">
           <h2>tick.dataset</h2>
        </a>
        <p>
            Provides easy access to datasets used as benchmarks in tick.
        </p>
     </div>
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/preprocessing.html">
           <h2>tick.preprocessing</h2>
        </a>
        <p>Some tools for preprocessing, such as features binarization and
            tools for preprocessing longitudinal features.
        </p>
     </div>
  </div>


  <!-- tick.metrics and tick.R and -->

  <div class="row">
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/metrics.html">
           <h2>tick.metrics</h2>
        </a>
        <p>
           Some tools computing specific metrics in tick.
        </p>
     </div>
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/R.html">
           <h2>Use tick in R</h2>
        </a>
        <p>
           How to use tick from the R software.
        </p>
     </div>
 </div>

  <!-- Dev and API -->

  <div class="row" style="margin-bottom:40px">
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/dev.html">
           <h2>Development</h2>
        </a>
        <p>
           You want to contribute ? Here you will find many tips.
        </p>
     </div>
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/api.html">
           <h2>API reference</h2>
        </a>
        <p>
           The full tick API
        </p>
     </div>
  </div>


.. toctree::
    :maxdepth: 2
    :hidden:

    modules/hawkes
    modules/linear_model

    modules/robust
    modules/survival

    modules/prox
    modules/solver

    modules/simulation
    modules/plot

    modules/dataset
    modules/preprocessing

    modules/metrics
    modules/R

    modules/dev
    modules/api
