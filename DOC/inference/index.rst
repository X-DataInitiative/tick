Inference
=========

These classes are called learners. They are meant to be very user friendly
and are most of the time good enough to infer many models.

These classes are scikit-learn compatible and, when it makes sense,
implement the following methods:

* `fit`
* `score`
* `predict`

========================================== ==========================================
Model                                      Class
========================================== ==========================================
:ref:`learner-logreg`                      `LogisticRegression`
:ref:`learner-hawkes-exp`                  `HawkesExpKern`
:ref:`learner-hawkes-sumexp`               `HawkesSumExpKern`
:ref:`learner-hawkes-conditional-law`      `HawkesConditionalLaw`
:ref:`learner-hawkes-em`                   `HawkesEM`
:ref:`learner-hawkes-adm4`                 `HawkesADM4`
:ref:`learner-hawkes-basis-kernels`        `HawkesBasisKernels`
:ref:`learner-hawkes-sumgaussians`         `HawkesSumGaussians`
========================================== ==========================================
