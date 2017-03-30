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
    ArrayDouble mu(iterate.size());
    ArrayDouble fixed_w = next_iterate;
    model->grad(fixed_w, mu);

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
                iterate[j] = iterate[j] - step * (grad_i[j] - grad_i_fixed_w[j] + mu[j]);
            }
            prox->call(iterate, step, iterate);

            if (variance_reduction == VarianceReductionMethod::Random && t == rand_index)
                next_iterate = iterate;

            if (variance_reduction == VarianceReductionMethod::Average)
                next_iterate.mult_incr(iterate, 1.0 / epoch_size);
        }

        if (variance_reduction == VarianceReductionMethod::Last)
            next_iterate = iterate;
    }

    t += epoch_size;
}

void SVRG::solve_sparse() {
    // TODO: once lazy updates will be implemented in prox we will be able to
    // do lazy updating with mu vector

    // The model is sparse, so it is a ModelGeneralizedLinear and the iteration looks a
    // little bit different
    ulong n_features = model->get_n_features();
    bool use_intercept = model->use_intercept();

    ArrayDouble mu(iterate.size());
    ArrayDouble fixed_w = iterate;
    model->grad(fixed_w, mu);

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
        // Sparse features vector
        BaseArrayDouble x_i = model->get_features(i);
        // Gradients factor
        double alpha_i_iterate = model->grad_i_factor(i, iterate);
        double alpha_i_fixed_w = model->grad_i_factor(i, fixed_w);
        double delta = -step * (alpha_i_iterate - alpha_i_fixed_w);
        if (use_intercept) {
            // Get the features vector, which is sparse here
            ArrayDouble iterate_no_interc = view(iterate, 0, n_features);
            //
            iterate_no_interc.mult_incr(x_i, delta);
            iterate[n_features] += delta;
            iterate.mult_incr(mu, -step);
        } else {
            iterate.mult_incr(x_i, delta);
            iterate.mult_incr(mu, -step);
        }

        prox->call(iterate, step, iterate);

        if (variance_reduction == VarianceReductionMethod::Random && t == rand_index)
            next_iterate = iterate;

        if (variance_reduction == VarianceReductionMethod::Average)
            next_iterate.mult_incr(iterate, 1.0 / epoch_size);
    }

    if (variance_reduction == VarianceReductionMethod::Last)
        next_iterate = iterate;
}

void SVRG::set_starting_iterate(ArrayDouble &new_iterate) {
    StoSolver::set_starting_iterate(new_iterate);

    next_iterate = iterate;
}
