.. title:: Tick

.. raw:: html

  <div
    class="jumbotron"
    style="padding-top: 10px; padding-bottom: 30px;"
  >
     <div class="container">
        <h1 style="font-size:40px">tick</h1>
        <p style="font-size:18px">
           tick is yet another machine learning library for Python 3, with
           a particular emphasis on point processes (Poisson processes,
           Hawkes processes, Cox regression), but includes also generalized
           linear models (GLM). It comes with optimization algorithms for
           inference and provides tools for simulation of datasets.
           A particular focus is on optimization: an extensive optimization
           toolbox is proposed, with recent state-of-the-art stochastic
           solvers.
        </p>
        <a class="btn btn-primary btn-lg" href="z_tutorials/index.html" role="button">
           Show me »
        </a>
     </div>
  </div>

  <div class="row">
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="z_tutorials/index.html">
           <h2>Tutorials</h2>
        </a>
        <p>
           Let us show you all the uses cases you might want to solve using
           tick. This includes simulation of many types, inference of
           different models or how to use the optimization toolbox.
        </p>
     </div>

     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="inference/index.html">
           <h2>Inference</h2>
        </a>
        <p>
           User friendly classes that are used to learn models from data.
           These classes are scikit-learn compatible.
        </p>
     </div>
  </div>

  <div class="row">
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="simulation/index.html">
           <h2>Simulation</h2>
        </a>
        <p>
           tick provides several classes to simulate datasets. This is
           particularly useful to test optimization algorithms, and to
           compare the statistical properties of inference methods.
        </p>
     </div>

     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="optim/index.html">
           <h2>Optimization</h2>
        </a>
        <p>
           This optimization toolbox is the core of most learners. It allows
           you to apply many optimization algorithms on a set of models and
           regularization techniques of your choice.
        </p>
     </div>
  </div>

  <div class="row" style="margin-bottom:40px">
     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="api/index.html">
           <h2>API reference</h2>
        </a>
        <p>
           A global vision of all tick's possibilities
        </p>
     </div>

     <div class="col-sm-4 col-sm-offset-1">
        <a class="reference internal" href="dev/index.html">
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

    z_tutorials/index
    inference/index
    simulation/index
    optim/index
    api/index
    dev/index

