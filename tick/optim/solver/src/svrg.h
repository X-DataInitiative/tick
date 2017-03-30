//
// Created by Martin Bompaire on 23/10/15.
//

#ifndef TICK_OPTIM_SOLVER_SRC_SVRG_H_
#define TICK_OPTIM_SOLVER_SRC_SVRG_H_

#include "array.h"
#include "sgd.h"
#include "../../prox/src/prox.h"

class SVRG : public StoSolver {
 public:
    enum class VarianceReductionMethod {
        Last    = 1,
        Average = 2,
        Random  = 3,
    };

 private:
    double step;
    VarianceReductionMethod variance_reduction;
    ArrayDouble next_iterate;

 public:
    SVRG(ulong epoch_size,
         double tol,
         RandType rand_type,
         double step,
         int seed = -1,
         VarianceReductionMethod variance_reduction = VarianceReductionMethod::Last);

    void solve() override;

    double get_step() const {
        return step;
    }

    void set_step(double step) {
        SVRG::step = step;
    }

    VarianceReductionMethod get_variance_reduction() const {
        return variance_reduction;
    }

    void set_variance_reduction(VarianceReductionMethod variance_reduction) {
        SVRG::variance_reduction = variance_reduction;
    }

    void set_starting_iterate(ArrayDouble &new_iterate) override;

    void solve_sparse();
};

#endif  // TICK_OPTIM_SOLVER_SRC_SVRG_H_
