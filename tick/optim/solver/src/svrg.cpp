//
// Created by Martin Bompaire on 23/10/15.
//

#include "svrg.h"

SVRG::SVRG(ulong epoch_size,
           double tol,
           RandType rand_type,
           double step,
           int seed,
           VarianceReductionMethod variance_reduction
)
    : StoSolver(epoch_size, tol, rand_type, seed),
      step(step), variance_reduction(variance_reduction) {
}


void SVRG::solve() {

    full_gradient = ArrayDouble(iterate.size());
    fixed_w = next_iterate;
    model->grad(fixed_w, full_gradient);

    if (model->is_sparse()) {
        solve_sparse();
    } else {
        // Dense case
        ArrayDouble grad_i(iterate.size());
        ArrayDouble grad_i_fixed_w(iterate.size());

        ulong rand_index{0};

        if (variance_reduction == VarianceReductionMethod::Random ||
            variance_reduction == VarianceReductionMethod::Average) {
            next_iterate.init_to_zero();
        }

        if (variance_reduction == VarianceReductionMethod::Random) {
            rand_index = rand_unif(epoch_size);
        }

        for (ulong t = 0; t < epoch_size; ++t) {
            ulong i = get_next_i();
            model->grad_i(i, iterate, grad_i);
            model->grad_i(i, fixed_w, grad_i_fixed_w);
            for (ulong j = 0; j < iterate.size(); ++j) {
                iterate[j] = iterate[j] - step * (grad_i[j] - grad_i_fixed_w[j] + full_gradient[j]);
            }
            prox->call(iterate, step, iterate);

            if (variance_reduction == VarianceReductionMethod::Random && t == rand_index)
                next_iterate = iterate;

            if (variance_reduction == VarianceReductionMethod::Average)
                next_iterate.mult_incr(iterate, 1.0 / epoch_size);

          // std::cout << "t= " << t << std::endl;
          // iterate.print();
        }

        if (variance_reduction == VarianceReductionMethod::Last)
            next_iterate = iterate;
    }

    t += epoch_size;

  // std::cout << "end epoch" << std::endl;
  // iterate.print();

}

void SVRG::solve_sparse() {
  // Data is sparse.
  // This means that model is a child of ModelGeneralizedLinear.
  // The iterations within an update will therefore look very different.
  // The strategy used here uses the delayed gradient and
  // penalization trick: we only work inside the current support
  // (non-zero values) of the sampled vector of features

  ulong n_features = model->get_n_features();
  bool use_intercept = model->use_intercept();

  // We need a copy of the current iterate, at which the full gradient will be computed
  // ArrayDouble fixed_w = iterate;
  // Computation of the full gradient once and for all in this epoch
  // model->grad(fixed_w, full_gradient);

  // The array will contain the iteration index of the last update of each
  // coefficient (model-weights and intercept)
  ArrayULong last_time(n_features);
  last_time.fill(0);

  ulong rand_index{0};

  if (variance_reduction == VarianceReductionMethod::Random ||
    variance_reduction == VarianceReductionMethod::Average) {
    next_iterate.init_to_zero();
  }

  if (variance_reduction == VarianceReductionMethod::Random) {
    rand_index = rand_unif(epoch_size);
  }

  for (t = 0; t < epoch_size; ++t) {
    // Get next sample index
    ulong i = get_next_i();
    // Sparse features vector
    BaseArrayDouble x_i = model->get_features(i);
    // Gradients factors (model is a GLM)
    double alpha_i_iterate = model->grad_i_factor(i, iterate);
    double alpha_i_fixed_w = model->grad_i_factor(i, fixed_w);
    double delta = alpha_i_iterate - alpha_i_fixed_w;

    // We update the iterate within the support of the features vector
    for (ulong idx_nnz = 0; idx_nnz < x_i.size_sparse(); ++idx_nnz) {
      // Get the index of the idx-th sparse feature of x_i
      ulong j = x_i.indices()[idx_nnz];
      // How many iterations since the last update of feature j

      //std::cout << "t= " << t << "last_time[j]= " << last_time[j] << std::endl;

      ulong delay_j = 0;
      if (t > last_time[j] + 1) {
        delay_j = t - last_time[j] - 1;
      }

      //std::cout << "delay_j= " << delay_j << std::endl;
      //
      double full_gradient_j = full_gradient[j];

      //std::cout << "full_gradient_j= " << full_gradient_j << std::endl;

      if(delay_j > 0) {
        // If there is delay, then we need to update coordinate j of the iterate first
        // We need to apply the delayed gradient steps for variance reduction
        iterate[j] -= step * delay_j * full_gradient_j;

       // std::cout << "step= " << step << std::endl;
        // And we need to apply the delayed regularization
        // std::cout << "idx_nnz= " << idx_nnz << std::endl;

        //std::cout << "iterate[j]= " << iterate[j] << std::endl;
        prox->_call_i(j, iterate, step, iterate, delay_j);
        //std::cout << "iterate[j]= " << iterate[j] << std::endl;
        // std::cout << "idx_nnz= " << idx_nnz << std::endl;
      }
      //std::cout << "idx_nnz= " << idx_nnz << std::endl;
      //std::cout << "iterate[j]= " << iterate[j] << std::endl;
      // Apply gradient descent to the model weights in the support of x_i
      iterate[j] -= step * (x_i.data()[idx_nnz] * delta + full_gradient_j);
      //std::cout << "iterate[j]= " << iterate[j] << std::endl;
      // std::cout << "idx_nnz= " << idx_nnz << std::endl;

      //std::cout << "iterate[j]= " << iterate[j] << std::endl;
      // Regularize the features of the model weights in the support of x_i
      prox->_call_i(j, iterate, step, iterate);
      //std::cout << "iterate[j]= " << iterate[j] << std::endl;
      // std::cout << "idx_nnz= " << idx_nnz << std::endl;

      // Update last_time
      last_time[j] = t;
      //std::cout << "t " << t << std::endl;

      // std::cout << "idx_nnz= " << idx_nnz << std::endl;

      // And let's not forget to update the intercept as well

      // std::cout << "idx_nnz= " << idx_nnz << std::endl;
    }
    if (use_intercept) {
      iterate[n_features] -= step * (delta + full_gradient[n_features]);
      // NB: no lazy-updating for the intercept, and no prox applied on it
    }

    if (variance_reduction == VarianceReductionMethod::Random && t == rand_index) {
      next_iterate = iterate;
    }

    if (variance_reduction == VarianceReductionMethod::Average) {
      next_iterate.mult_incr(iterate, 1.0 / epoch_size);
    }

    //std::cout << "t= " << t << std::endl;
    //iterate.print();
  }

  //std::cout << "end epoch" << std::endl;
  //std::cout << "t= " << t << std::endl;

  // Now we need to fully update the iterate (not the intercept),
  // since we reached the end of the epoch
  for(ulong j=0; j < n_features; ++j) {
    ulong delay_j = 0;
    if (t > last_time[j] + 1) {
      delay_j = t - last_time[j] - 1;
    }

    //std::cout << "end epoch" << std::endl;
    //std::cout << "t= " << t << "last_time[j]= " << last_time[j] << std::endl;
    //std::cout << "delay_j= " << delay_j << "full_gradient[j]= " << full_gradient[j] << std::endl;

    if(delay_j > 0) {
      // If there is delay, then we need to update coordinate j of the iterate first
      // We need to apply the delayed gradient steps for variance reduction
      iterate[j] -= step * delay_j * full_gradient[j];
      // And we need to apply the delayed regularization
      prox->_call_i(j, iterate, step, iterate, delay_j);
    }

    // iterate.print();
  }

  t += epoch_size;



  if (variance_reduction == VarianceReductionMethod::Last)
      next_iterate = iterate;
}

void SVRG::set_starting_iterate(ArrayDouble &new_iterate) {
    StoSolver::set_starting_iterate(new_iterate);
    next_iterate = iterate;
}


/*

void SVRG::solve_sparse() {
    ulong n_coeffs = model->get_n_coeffs();
    ulong n_features = model->get_n_features();
    bool use_intercept = model->use_intercept();

    // Data is sparse. This means that model is a ModelGeneralizedLinear.
    // The array will contain the iteration index of the last update of each
    // coefficient (model-weights and intercept)
    ArrayULong last_time(n_features);
    last_time.fill(0);

    // An array for the full gradient used in variance reduction
    ArrayDouble full_gradient(n_coeffs);
    // We need a copy of the current iterate, at which the full gradient used
    // for variance reduction is computed, and used all along this epoch.
    ArrayDouble fixed_w = iterate;
    // We compute the full gradient once in this epoch
    model->grad(fixed_w, full_gradient);

    for (ulong t = 0; t < epoch_size; ++t) {
        // Get next sample index
        ulong i = get_next_i();
        // Get the sparse features vector
        BaseArrayDouble x_i = model->get_features(i);
        // Compute gradients factors (model is a GLM)
        // TODO: could be done at the same time (within the same loop) reduces 2 * s
        // TODO: complexity to s (where s is number of non-zeros in x_i)
        double alpha_i_iterate = model->grad_i_factor(i, iterate);
        double alpha_i_fixed_w = model->grad_i_factor(i, fixed_w);
        double delta = alpha_i_iterate - alpha_i_fixed_w;

        // We need to correct the current iterate in the support (gradient steps
        // and penalization are delayed)

        // We update the iterate within the support of the features vector
        for (ulong idx_nnz = 0; idx_nnz < x_i.size_sparse(); idx_nnz++) {
            // Get the index of the idx-th sparse feature of x_i
            ulong j = x_i.indices()[idx_nnz];
            // How many iterations since the last update of feature j
            ulong delay_j = t - last_time[j] - 1;
            //
            double full_gradient_j = full_gradient[j];

            if(delay_j > 0) {
                // If there is delay, then we need to update coordinate j of the iterate first
                // We need to apply the delayed gradient steps for variance reduction
                iterate[j] -= step * delay_j * full_gradient_j;
                // And we need to apply the delayed regularization
                prox->_call_i(j, iterate, step, iterate, delay_j);
            }
            // Apply gradient descent to the model weights in the support of x_i
            iterate[j] -= step * (x_i.data()[idx_nnz] * delta + full_gradient_j);
            // Regularize the features of the model weights in the support of x_i
            prox->_call_i(j, iterate, step, iterate);
            // Update last_time
            last_time[j] = t;
        }
        // And let's not forget to update the intercept as well
        if (use_intercept) {
            iterate[n_features] -= step * (delta + full_gradient[n_features]);
            // NB: no lazy-updating for the intercept, and no prox applied on it
        }
    }

    // Now we need to fully update the iterate (not the intercept),
    // since we reached the end of the epoch
    for(ulong j=0; j < n_features; ++j) {
        ulong delay_j = t - last_time[j] - 1;
        if(delay_j > 0) {
            // If there is delay, then we need to update coordinate j of the iterate first
            // We need to apply the delayed gradient steps for variance reduction
            iterate[j] -= step * delay_j * full_gradient[j];
            // And we need to apply the delayed regularization
            prox->_call_i(j, iterate, step, iterate, delay_j);
        }
    }
}
*/
