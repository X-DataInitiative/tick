//
// Created by Stéphane GAIFFAS on 12/04/2016.
//

#ifndef TICK_OPTIM_MODEL_SRC_COXREG_PARTIAL_LIK_H_
#define TICK_OPTIM_MODEL_SRC_COXREG_PARTIAL_LIK_H_

// License: BSD 3 clause

#include "model.h"


class DLL_PUBLIC ModelCoxRegPartialLik : public Model {
 private:
    ArrayDouble inner_prods;
    ArrayDouble s1;
    ArrayULong idx;

 protected:
    ulong n_samples, n_features, n_failures;

    SBaseArrayDouble2dPtr features;
    ArrayDouble times;
    ArrayUShort censoring;
    ArrayULong idx_failures;

    inline BaseArrayDouble get_feature(ulong i) const {
        return view_row(*features, idx[i]);
    }

    inline double get_time(ulong i) const {
        return times[i];
    }

    inline ushort get_censoring(ulong i) const {
        return censoring[i];
    }

    inline ulong get_idx_failure(ulong i) const {
        return idx_failures[i];
    }

 public:
    ModelCoxRegPartialLik(const SBaseArrayDouble2dPtr features,
                          const SArrayDoublePtr times,
                          const SArrayUShortPtr censoring);

    const char *get_class_name() const override {
        return "ModelCoxRegPartialLik";
    }

    /**
     * \brief Computation of the value of minus the partial Cox
     * log-likelihood at
     * point coeffs.
     * It should be overflow proof and fast.
     *
     * \note
     * This code assumes that the times are inversely sorted and that the
     * rows of the features matrix and index of failure times are sorted
     * accordingly. This sorting is done automatically by the SurvData object.
     *
     * \param coeffs : The vector at which the loss is computed
    */
    double loss(const ArrayDouble &coeffs) override;

    void grad(const ArrayDouble &coeffs, ArrayDouble &out) override;
};


typedef std::shared_ptr<ModelCoxRegPartialLik> ModelCoxRegPartialLikPtr;


#endif  // TICK_OPTIM_MODEL_SRC_COXREG_PARTIAL_LIK_H_
