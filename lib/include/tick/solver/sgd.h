//
// Created by Martin Bompaire on 22/10/15.
//

#ifndef LIB_INCLUDE_TICK_SOLVER_SGD_H_
#define LIB_INCLUDE_TICK_SOLVER_SGD_H_

// License: BSD 3 clause

#include "tick/base_model/model.h"
#include "tick/prox/prox.h"
#include "sto_solver.h"

class SGD : public StoSolver {
 private:
    double step_t;
    double step;

 public:
    SGD(ulong epoch_size = 0,
        double tol = 0.,
        RandType rand_type = RandType::unif,
        double step = 0.,
        int seed = -1);

    inline double get_step_t() const {
        return step_t;
    }

    inline double get_step() const {
        return step;
    }

    inline void set_step(double step) {
        this->step = step;
    }

    void solve();

    void solve_sparse();

    inline double get_step_t();
};

#endif  // LIB_INCLUDE_TICK_SOLVER_SGD_H_
