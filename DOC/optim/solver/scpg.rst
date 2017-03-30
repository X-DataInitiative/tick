:orphan:

.. _scpg:


Self Concordant Proximal Gradient
---------------------------------

Self Concordant Proximal Gradient. An iteration of the
algorithm is given by

.. math::
    \DeclareMathOperator{\prox}{prox}

    \begin{align*}
    y_{k} &= \prox_{g/L} \left(x_{k} - \frac{1}{L_k} \nabla f(x_{k}) \right) \\
    d_k &= y_k - x_k \\
    \beta_k &= \sqrt{L_k} \| d_k \|_2 \\
    \lambda_k &= \sqrt{d_k^T \nabla^2 f(x_k) d_k } \\
    x_{k+1} &= x_k + \alpha_k d_k = \alpha_k y_k + x_k (1 - \alpha_k) \\
    \end{align*}

with
:math:`\alpha_k = \frac{\beta_k^2}{\lambda_k(\lambda_k + \beta_k^2)} \in ]0, 1]`
is the step size.

This algorithm relies on the fact that the given model is self-concordant

Tran-Dinh, Quoc, Anastasios Kyrillidis, and Volkan Cevher.
"Composite self-concordant minimization."
arXiv preprint arXiv:1308.2867 (2013).

.. autoclass:: tick.optim.solver.SCPG
    :members:
    :inherited-members:
