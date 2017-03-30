Hawkes with exponential kernels and known decays
================================================

Hawkes processes are point processes defined by the intensity:

.. math::

    \forall i \in [1 \dots D], \quad
    \lambda_i(t) = \mu_i + \sum_{j=1}^D
    \sum_{t_k^j < t} \phi_{ij}(t - t_k^j)

where

* :math:`D` is the number of nodes
* :math:`\mu_i` are the baseline intensities
* :math:`\phi_{ij}` are the kernels
* :math:`t_k^j` are the timestamps of all events of node :math:`j`

In this case we are interested in the following exponential parametrisation of
the kernels:

.. math::
    \phi_{ij}(t) = \alpha^{ij} \beta^{ij}
                   \exp (- \beta^{ij} t) 1_{t > 0}

where

* Matrix :math:`A = (\alpha^{ij})_{ij} \in \mathbb{R}^{D \times D}`
  is the adjacency matrix
* Matrix :math:`B = (\beta_{ij})_{ij} \in \mathbb{R}^{D \times D}` is the
  decay matrix. This parameter is given to the model

This parametrization is useful as it allows to do very fast inference.

Basic optimization
------------------

We can use learner `HawkesExpKern` to infer these Hawkes models. We
provide here a quick example. An exhaustive list of its capabilities is
available in the class documentation (`tick.inference.HawkesExpKern`)

.. plot:: z_tutorials/inference/code_samples/hawkes_matrix_exp_kernels.py
    :include-source:

Hawkes ADM4 for lasso and nuclear penalization
----------------------------------------------

In order to obtain sparse and low-rank adjacency matrix, it might be
efficient to perform a mix of lasso and nuclear penalization. This specific
combination is described in

Zhou, K., Zha, H., & Song, L. (2013, May).
Learning Social Infectivity in Sparse Low-rank Networks Using
Multi-dimensional Hawkes Processes. In `AISTATS (Vol. 31, pp. 641-649)
<http://www.jmlr.org/proceedings/papers/v31/zhou13a.pdf>`_.

and we have implemented it as `HawkesADM4` (`tick.inference.HawkesADM4`).

.. plot:: z_tutorials/inference/code_samples/hawkes_adm4.py
    :include-source:

Note that `HawkesADM4` can only be used with one decay :math:`\beta` shared
by all exponential kernels.

Sum of exponential kernels
--------------------------

If kernels does not have an exact exponential shape, it is possible to
approximate more shapes with sum of exponentials kernels. These have the
following parametrization:

.. math::
    \phi_{ij}(t) = \sum_{u=1}^{U} \alpha^u_{ij} \beta^u
                   \exp (- \beta^u t) 1_{t > 0}

where

* :math:`U` is the number of exponential decays
* Matrix :math:`A = (\alpha^u_{ij})_{ij} \in \mathbb{R}^{D \times D
  \times U}` is the adjacency matrix
* Vector :math:`\beta \in \mathbb{R}^{U}` denotes the exponentials decays.
  This parameter is given to the model.

We can use learner `HawkesSumExpKern` to infer these Hawkes models. We
provide here a quick example. An exhaustive list of its capabilities is
available in the class documentation (`tick.inference.HawkesSumExpKern`)


.. plot:: z_tutorials/inference/code_samples/hawkes_sum_exp_kernels.py
    :include-source:

