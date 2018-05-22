// License: BSD 3 clause

#include "tick/solver/adagrad.h"
#include "tick/prox/prox_separable.h"

template <class T, class K>
TAdaGrad<T, K>::TAdaGrad(ulong epoch_size, T tol, RandType rand_type, T step,
                         int seed)
    : TStoSolver<T, K>(epoch_size, tol, rand_type, seed),
      hist_grad(iterate.size()),
      step(step) {}

template <class T, class K>
void TAdaGrad<T, K>::solve() {
  std::shared_ptr<TProxSeparable<T, K>> casted_prox;
  if (prox->is_separable()) {
    casted_prox = std::static_pointer_cast<TProxSeparable<T, K>>(prox);
  } else {
    TICK_ERROR("Prox in Adagrad must be separable but got "
               << prox->get_class_name());
  }

  Array<T> grad_i(iterate.size());
  grad_i.init_to_zero();

  Array<T> steps(iterate.size());

  const ulong prox_start = prox->get_start();
  const ulong prox_end = prox->get_end();

  const ulong start_t = t;
  for (t = start_t; t < start_t + epoch_size; ++t) {
    const ulong i = get_next_i();
    model->grad_i(i, iterate, grad_i);

    for (ulong j = 0; j < grad_i.size(); ++j) {
      hist_grad[j] += grad_i[j] * grad_i[j];
    }

    // We add this constant in case the sqrt below approaches 0.0
    const T jitter = 1e-6;

    for (ulong j = 0; j < hist_grad.size(); ++j) {
      steps[j] = step / (std::sqrt(hist_grad[j] + jitter));
    }

    for (ulong j = 0; j < iterate.size(); ++j) {
      iterate[j] = iterate[j] - steps[j] * grad_i[j];
    }

    Array<T> prox_steps = view(steps, prox_start, prox_end);

    casted_prox->call(iterate, prox_steps, iterate);
  }
}

template <class T, class K>
void TAdaGrad<T, K>::set_starting_iterate(Array<T> &new_iterate) {
  TStoSolver<T, K>::set_starting_iterate(new_iterate);

  hist_grad = Array<T>(new_iterate.size());
  hist_grad.init_to_zero();
}

template class DLL_PUBLIC TAdaGrad<double>;
template class DLL_PUBLIC TAdaGrad<float>;
