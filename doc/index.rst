.. title:: Tick

.. raw:: html

  <div
    class="jumbotron"
    style="padding-top: 10px; padding-bottom: 30px;"
  >
     <div class="container">
        <h1 style="font-size:40px">tick</h1>
        <p style="font-size:18px">
           tick a machine learning library for Python 3.
           The focus is on statistical learning for time dependent systems,
           such as point processes. Tick features also tools for generalized
           linear models, and a generic optimization toolbox.
        </p>
        <p style="font-size:18px">
           The core of the library is an optimization module providing model
           computational classes, solvers and proximal operators for regularization.

           It comes also with inference and simulation tools intended for end-users.
        </p>
        <a class="btn btn-primary btn-lg" href="auto_examples/index.html"
        role="button">
           Show me »
        </a>
     </div>
  </div>

  <div class="row">
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="auto_examples/index.html">
           <h2>Examples</h2>
        </a>
        <p>
           Examples of how to simulate models, use the optimization toolbox, or
           use user-friendly inference tools.
        </p>
     </div>

     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/inference.html">
           <h2>Inference</h2>
        </a>
        <p>
           User-friendly classes for inference of models
        </p>
     </div>
  </div>

  <div class="row">
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/simulation.html">
           <h2>Simulation</h2>
        </a>
        <p>
            User-friendly classes for simulation of data
        </p>
     </div>

     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/optim.html">
           <h2>Optimization</h2>
        </a>
        <p>
           The core module of the library: an optimization toolbox
           consisting of models, solvers and prox (penalization) classes.
           Almost all of them can be combined together.
        </p>
     </div>
  </div>

  <div class="row" style="margin-bottom:40px">
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/api.html">
           <h2>API reference</h2>
        </a>
        <p>
           The full tick API
        </p>
     </div>

     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="modules/dev.html">
           <h2>Development</h2>
        </a>
        <p>
           You would like to contribute? Here you will find many tips.
        </p>
     </div>
  </div>


.. toctree::
    :maxdepth: 2
    :hidden:

    modules/inference
    modules/optim
    modules/plot
    modules/preprocessing
    modules/simulation
    modules/dataset
    modules/dev
