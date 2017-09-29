#include "base.h"
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
#include <float.h>


void proj_simplex(ArrayDouble & coeffs, ArrayDouble & out, double r) {
    // Projection onto the simplex or radius r
    // It seems to be OK
    unsigned long length, nnz = 0;
    double coeffs_i, out_i, z_i, z_nnz_minus_1, coeffs_i_minus_z_nnz_minus_1, cumsum_out = 0;
    double s = 0;
    bool test = false;

    length = coeffs.size();
    ArrayDouble z = ArrayDouble(length);

    // First we check if coeffs is in the simplex of radius r
    for(unsigned long i=0; i < length; ++i) {
        coeffs_i = coeffs[i];
        if(coeffs_i < 0) {
            test = true;
        }
        s += coeffs_i;
    }
    if(s > r) {
        test = true;
    }
    if(test) {
        // Now the code assumes that out contains a sorted copy of coeffs in reverse order
        for(unsigned long i=0; i < length; ++i) {
            out_i = out[i];
            cumsum_out += out_i;
            z_i = (cumsum_out - r) / (i + 1);
            z[i] = z_i;
            if(out_i > z_i) {
                nnz++;
            }
        }
        // get the z before the hitting point
        z_nnz_minus_1 = z[nnz - 1];
        // project
        for(unsigned long i=0; i < length; ++i) {
           coeffs_i = coeffs[i];
            coeffs_i_minus_z_nnz_minus_1 = coeffs_i - z_nnz_minus_1;
            if(coeffs_i_minus_z_nnz_minus_1 > 0) {
                out[i] = coeffs_i_minus_z_nnz_minus_1;
            } else {
                out[i] = 0;
            }
        }
    } else {
        // No need for projection, just put pack the coeff
        // TODO: this is stupid, think about it better...
        for(unsigned long i=0; i < length; ++i) {
            out[i] = coeffs[i];
        }
    }
}


unsigned int proj_half_spaces(ArrayDouble & coeffs, ArrayDouble2d &A, ArrayDouble &b,
                              ArrayDouble &norms, ArrayDouble &out,
                              const unsigned int max_pass,
                              ArrayDouble &history) {
    // Projection onto an intersection of half-spaces, using the cycling method (add reference)
    // The code assumes that out contains a copy of coeffs

    unsigned long n_constraints = A.n_rows();
    unsigned long n_coeffs = A.n_cols();
    // Maximum number of passes over the constraints
    unsigned int max_passes = 100;
    double d_i;
    unsigned long n_ok;
    unsigned int n_pass = 0;
    const double MAX_DOUBLE = std::numeric_limits<double>::max();
    // Worst violation of the constraint
    double min_di = std::numeric_limits<double>::min();
    for(n_pass=0; n_pass < max_passes; n_pass++) {
        // Loop over the constraints until all of them are satisfied
        n_ok = 0;
        min_di = MAX_DOUBLE;
        for(unsigned long i=0; i < n_constraints; i++) {
            // d_i = a_i^T x - b_i
            ArrayDouble a_i = view_row(A,i);
            d_i = a_i.dot(out) - b[i];
            if (d_i <= min_di) {
                // Keep the minimum of the d_i
                min_di = d_i;
            }
            if(d_i >= 0) {
                // The constraint is satisfied
                n_ok++;
            } else {
                // The constraint is not satisfied: project
                for(unsigned long j=0; j < n_coeffs; j++) {
                    out[j] -= d_i / norms[i] * a_i[j];
                }
            }
        }
        history[n_pass] = min_di;
        if(n_ok == n_constraints) {
            // All constraints are satisfied
            break;
        }
    }
    return n_pass;
}

