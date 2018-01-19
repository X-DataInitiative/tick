//
// Created by Martin Bompaire on 22/10/15.
//

#ifndef LIB_INCLUDE_TICK_SOLVER_STO_SOLVER_H_
#define LIB_INCLUDE_TICK_SOLVER_STO_SOLVER_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "tick/random/rand.h"
#include "tick/base_model/model.h"
#include "tick/prox/prox.h"


// TODO: code an abstract class and use it for StoSolvers
// TODO: StoSolver and LabelsFeaturesSolver


// Type of randomness used when sampling at random data points
enum class RandType {
    unif = 0,
    perm
};

// Base abstract for a stochastic solver
class StoSolver {
 protected:
    // Model object
    ModelPtr model;

    ProxPtr prox;

    Rand rand;

    // Iteration counter
    ulong t = 1;

    // Iterate
    ArrayDouble iterate;

    // sampling is done in {0, ..., rand_max-1}
    // This is useful to know in what range random sampling must be done
    // (n_samples for generalized linear models, n_nodes for Hawkes processes, etc.)
    ulong rand_max;

    // Number of steps within an epoch
    ulong epoch_size;

    // Tolerance for convergence. Not used yet.
    double tol;

    // Type of random sampling
    RandType rand_type;

    // An array that allows to store the sampled random permutation
    ArrayULong permutation;

    // Current index in the permutation (useful when using random permutation sampling)
    ulong i_perm;

    // A flag that specify if random permutation is ready to be used or not
    bool permutation_ready;

    // Init permutation array in case of Random is srt to permutation
    void init_permutation();

    // Seed of the random sampling
    int seed;

 public:
    explicit StoSolver(int seed = -1);

    StoSolver(ulong epoch_size = 0,
              double tol = 0.,
              RandType rand_type = RandType::unif,
              int seed = -1);

    virtual ~StoSolver() = default;

    virtual void set_model(ModelPtr model) {
        this->model = model;
        permutation_ready = false;
        iterate = ArrayDouble(model->get_n_coeffs());
        iterate.init_to_zero();
    }

    virtual void set_prox(ProxPtr prox) {
        this->prox = prox;
    }

    void set_seed(int seed) {
        this->seed = seed;
        rand = Rand(seed);
    }

    virtual void reset();

    ulong get_next_i();

    void shuffle();

    virtual void solve() {}

    virtual void get_minimizer(ArrayDouble &out);

    virtual void get_iterate(ArrayDouble &out);

    virtual void set_starting_iterate(ArrayDouble &new_iterate);

    // Returns a uniform integer in the set {0, ..., m - 1}
    inline ulong rand_unif(ulong m) {
        return rand.uniform_int(ulong{0}, m);
    }

    inline double get_tol() const {
        return tol;
    }

    inline void set_tol(double tol) {
        this->tol = tol;
    }

    inline ulong get_epoch_size() const {
        return epoch_size;
    }

    inline void set_epoch_size(ulong epoch_size) {
        this->epoch_size = epoch_size;
    }

    inline ulong get_t() const {
        return t;
    }

    inline RandType get_rand_type() const {
        return rand_type;
    }

    inline void set_rand_type(RandType rand_type) {
        this->rand_type = rand_type;
    }

    inline ulong get_rand_max() const {
        return rand_max;
    }

    inline void set_rand_max(ulong rand_max) {
        this->rand_max = rand_max;
        permutation_ready = false;
    }
};

#endif  // LIB_INCLUDE_TICK_SOLVER_STO_SOLVER_H_
