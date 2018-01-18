// License: BSD 3 clause


#include "hawkes_fixed_sumexpkern_loglik_custom.h"

ModelHawkesSumExpCustom::ModelHawkesSumExpCustom(const ArrayDouble _decays, const ulong _MaxN_of_f, const int max_n_threads) :
        MaxN_of_f(_MaxN_of_f), ModelHawkesFixedKernLogLik(max_n_threads), decays(_decays) {}

void ModelHawkesSumExpCustom::allocate_weights() {
  if (n_nodes == 0) {
    TICK_ERROR("Please provide valid timestamps before allocating weights")
  }

    Total_events = n_total_jumps;
    ulong U = decays.size();

    g = ArrayDouble2dList1D(n_nodes);
    G = ArrayDouble2dList1D(n_nodes);
    sum_G = ArrayDoubleList1D(n_nodes);

    H1 = ArrayDoubleList1D(n_nodes);
    H2 = ArrayDoubleList1D(n_nodes);
    H3 = ArrayDoubleList1D(n_nodes);

    for (ulong i = 0; i != n_nodes; i++) {
        //0 + events + T
        g[i] = ArrayDouble2d(Total_events + 2, n_nodes * U);
        g[i].init_to_zero();
        G[i] = ArrayDouble2d(Total_events + 2, n_nodes * U);
        G[i].init_to_zero();
        sum_G[i] = ArrayDouble(n_nodes * U);
        sum_G[i].init_to_zero();

        H1[i] = ArrayDouble(MaxN_of_f);
        H1[i].init_to_zero();
        H2[i] = ArrayDouble(MaxN_of_f);
        H2[i].init_to_zero();
        H3[i] = ArrayDouble(MaxN_of_f * U);
        H3[i].init_to_zero();
    }
    global_timestamps = ArrayDouble(Total_events + 1);
    global_timestamps.init_to_zero();
    type_n = ArrayULong(Total_events + 1);
    type_n.init_to_zero();
    //global_n = ArrayLong(Total_events + 1);
    //global_n.init_to_zero();
}

void ModelHawkesSumExpCustom::compute_weights_dim_i(const ulong i) {
    ArrayDouble2d g_i = view(g[i]);
    ArrayDouble2d G_i = view(G[i]);

    //! hacked code here, seperator = 1, meaning L(increasing) is timestamps[1], C,M(decreasing) are timestamps[2] timestamps[3]
    ArrayULong tmp_pre_type_n(Total_events + 1);
    tmp_pre_type_n[0] = 0;
    ArrayULong tmp_index(Total_events + 1);

    ulong count = 1;
    for (ulong j = 0; j != n_nodes; j++) {
        const ArrayDouble t_j = view(*timestamps[j]);
        for (ulong k = 0; k != (*n_jumps_per_node)[j]; ++k) {
          global_timestamps[count] = t_j[k];
          tmp_pre_type_n[count++] = j + 1;
        }
    }

    global_timestamps.sort(tmp_index);

    for (ulong k = 1; k != Total_events + 1; ++k)
        type_n[k] = tmp_pre_type_n[tmp_index[k]];

    auto get_index = [=](ulong k, ulong j, ulong u) {
        return n_nodes * get_n_decays() * k + get_n_decays() * j + u;
    };

    ulong U = decays.size();
    for(ulong u = 0; u != U; ++u) {
        double decay = decays[u];

        //for the g_j, I make all the threads calculating the same g in this developing stage
        for (ulong j = 0; j != n_nodes; j++) {
            //! here k starts from 1, cause g(t_0) = G(t_0) = 0
            // 0 + Totalevents + T
            for (ulong k = 1; k != 1 + Total_events + 1; k++) {
                const double t_k = (k != (1 + Total_events) ? global_timestamps[k] : end_time);
                const double ebt = std::exp(-decay * (t_k - global_timestamps[k - 1]));
                if (k != 1 + Total_events)
                    g_i[get_index(k, j, u)] =
                            g_i[get_index(k - 1, j, u)] * ebt + (type_n[k - 1] == j + 1 ? decay * ebt : 0);
                G_i[get_index(k, j, u)] =
                        (1 - ebt) / decay * g_i[get_index(k - 1, j, u)] + ((type_n[k - 1] == j + 1) ? 1 - ebt : 0);

                // ! in the G, we calculated the difference between G without multiplying f
                // sum_G is calculated later, in the calculation of L_dim_i and its grads
            }
        }
    }

  //! in fact, H1, H2 is one dimension, here I make all threads calculate the same thing
  ArrayDouble H1_i = view(H1[i]);
  ArrayDouble H2_i = view(H2[i]);
  ArrayDouble H3_i = view(H3[i]);
  H1_i[global_n[Total_events]] -= 1;
  for (ulong k = 1; k != 1 + Total_events + 1; k++) {
      if (k != (1 + Total_events))
          H1_i[global_n[k - 1]] += (type_n[k] == i + 1 ? 1 : 0);
      const double t_k = (k != (1 + Total_events) ? global_timestamps[k] : end_time);
      H2_i[global_n[k - 1]] -= t_k - global_timestamps[k - 1];
  }

    for(ulong u = 0; u != U; ++u) {
        double decay = decays[u];
        for (ulong k = 1; k != 1 + Total_events + 1; k++) {
            const double t_k = (k != (1 + Total_events) ? global_timestamps[k] : end_time);
            //! recall that all g_i_j(t) are same, for any i
            //! thread_i calculate H3_i
            H3_i[global_n[k - 1] * U + u] -= G_i[get_index(k, i, u)];
        }
    }
}

ulong ModelHawkesSumExpCustom::get_n_coeffs() const {
  //!seems not ever used in this stage
  ulong U = decays.size();
  return n_nodes + n_nodes * n_nodes * U + n_nodes * MaxN_of_f;
}

void ModelHawkesSumExpCustom::set_data(const SArrayDoublePtrList1D &_timestamps,
                                 const SArrayLongPtr _global_n,
                                 const double _end_times){
  ModelHawkesSingle::set_data(_timestamps, _end_times);

  global_n = ArrayLong(n_total_jumps + 1);
  for(ulong k = 0; k != n_total_jumps + 1; ++k)
    global_n[k] = _global_n->value(k);
}

double ModelHawkesSumExpCustom::loss_dim_i(const ulong i,
                                                    const ArrayDouble &coeffs) {
    const double mu_i = coeffs[i];
    const ArrayDouble f_i = view(coeffs, get_f_i_first_index(i), get_f_i_last_index(i));
    double loss = 0;

    ulong U = this->decays.size();
    //cozy at hand
    const ArrayDouble2d g_i = view(g[i]);
    const ArrayDouble2d G_i = view(G[i]);

    auto get_index = [=](ulong k, ulong j, ulong u) {
        return n_nodes * decays.size() * k + decays.size() * j + u;
    };

    //term 1
    //end_time is T
    for (ulong k = 1; k != Total_events + 1; ++k)
        //! insert event t0 = 0 in the Total_events and global_n
        if (type_n[k] == i + 1)
            loss += log(f_i[global_n[k - 1]]);

    //term 2
    for (ulong k = 1; k != Total_events + 1; ++k)
        if (type_n[k] == i + 1) {
            double tmp_s = mu_i;
            for(ulong j = 0; j != n_nodes; ++j)
                for (ulong u = 0; u != U; ++u) {
                    double alpha_u_i_j = coeffs[get_alpha_u_i_j_index(u, i, j)];
                    tmp_s += alpha_u_i_j * g_i[get_index(k, j, u)];
                }

            if (tmp_s <= 0) {
                TICK_ERROR("The sum of the influence on someone cannot be negative. "
                                   "Maybe did you forget to add a positive constraint to "
                                   "your proximal operator, in SumExp::loss_dim_i");
            }
            loss += log(tmp_s);
        }

    //term 3,4
    for (ulong k = 1; k != Total_events + 1; k++)
        loss -= mu_i * (global_timestamps[k] - global_timestamps[k - 1]) * f_i[global_n[k - 1]];
    loss -= mu_i * (end_time - global_timestamps[Total_events]) * f_i[global_n[Total_events]];

    //! clean sum_G each time
    sum_G[i].init_to_zero();

    //term 5, 6
    //! sum_g already takes care of the last item T
    for (ulong j = 0; j != n_nodes; j++) {
        for (ulong u = 0; u != U; ++u) {
            double alpha_u_i_j = coeffs[get_alpha_u_i_j_index(u, i, j)];
            double sum_G_i_j_u = 0;
            for (ulong k = 1; k != 1 + Total_events + 1; k++) {
                sum_G_i_j_u += G_i[get_index(k, j, u)] * f_i[global_n[k - 1]];
            }
            loss -= alpha_u_i_j * sum_G_i_j_u;
        }
    }

    return -end_time - loss;
}

void ModelHawkesSumExpCustom::grad_dim_i(const ulong i,
                                                  const ArrayDouble &coeffs,
                                                  ArrayDouble &out) {
    const double mu_i = coeffs[i];
    double &grad_mu_i = out[i];

    ulong U = decays.size();
    auto get_index = [=](ulong k, ulong j, ulong u) {
        return n_nodes * decays.size() * k + decays.size() * j + u;
    };

    const ArrayDouble f_i = view(coeffs, get_f_i_first_index(i), get_f_i_last_index(i));
    ArrayDouble grad_f_i = view(out, get_f_i_first_index(i), get_f_i_last_index(i));

    //necessary information required
    const ArrayDouble2d g_i = view(g[i]);
    const ArrayDouble2d G_i = view(G[i]);

    //! grad of mu_i
    grad_mu_i = 0;
    for (ulong k = 1; k < Total_events + 1; ++k) {
        int tmp_flag = 0;
        if (k == Total_events + 1)
            tmp_flag = 1;
        else if (type_n[k] == i + 1)
            tmp_flag = 1;
        if (tmp_flag) {
            //! recall that all g_i are the same
            double denominator = mu_i;
            for (ulong j = 0; j != n_nodes; j++)
                for (ulong u = 0; u != U; ++u) {
                    double alpha_u_i_j = coeffs[get_alpha_u_i_j_index(u, i, j)];
                    denominator += alpha_u_i_j * g_i[get_index(k, j, u)];
                }
            grad_mu_i += 1 / denominator;
        }
    }

    for (ulong k = 1; k < 1 + Total_events + 1; k++) {
        const double t_k = (k != (Total_events + 1)) ? global_timestamps[k] : end_time;
        grad_mu_i -= (t_k - global_timestamps[k - 1]) * f_i[global_n[k - 1]];
    }

    //! grad of alpha_u_{ij}
    //! here we calculate the grad of alpha_ij_u, for all j and all u
    for (ulong k = 1; k < 1 + Total_events + 1; ++k) {
        int tmp_flag = 0;
        if (k == Total_events + 1)
            tmp_flag = 1;
        else if (type_n[k] == i + 1)
            tmp_flag = 1;
        if (tmp_flag) {
            //calculate the denominator
            double denominator = mu_i;
            for (ulong jj = 0; jj != n_nodes; ++jj)
                for (ulong uu = 0; uu != U; ++uu) {
                    double alpha_uu_i_jj = coeffs[get_alpha_u_i_j_index(uu, i, jj)];
                    denominator += alpha_uu_i_jj * g_i[get_index(k, jj, uu)];
                }
            for (ulong j = 0; j != n_nodes; ++j)
                for (ulong u = 0; u != U; ++u) {
                    double &grad_alpha_u_ij = out[get_alpha_u_i_j_index(u, i, j)];
                    grad_alpha_u_ij += g_i[get_index(k, j, u)] / denominator;
                }
        }
    }

    for (ulong j = 0; j != n_nodes; ++j)
        for (ulong u = 0; u != U; ++u) {
            double sum_G_i_j_u = 0;
            for (ulong k = 1; k != 1 + Total_events + 1; k++) {
                sum_G_i_j_u += G_i[get_index(k, j, u)] * f_i[global_n[k - 1]];
            }
            double &grad_alpha_u_ij = out[get_alpha_u_i_j_index(u, i, j)];
            grad_alpha_u_ij -= sum_G_i_j_u;
        }

    //! grad of f^i_n
    //! in fact, H1_i for different i keep the same information, same thing for H2, H3
    const ArrayDouble H1_i = view(H1[i]);
    const ArrayDouble H2_i = view(H2[i]);
    for (ulong n = 0; n != MaxN_of_f; ++n) {
        double result_dot = 0; //! alpha_i_j_u.dot(H3_j_u_n);
        for (ulong j = 0; j != n_nodes; ++j) {
            const ArrayDouble H3_j = view(H3[j]);
            for (ulong u = 0; u != U; ++u) {
                double alpha_u_i_j = coeffs[get_alpha_u_i_j_index(u, i, j)];
                result_dot += alpha_u_i_j * H3_j[n * U + u];
            }
        }
        grad_f_i[n] = H1_i[n] / f_i[n] + mu_i * H2_i[n] + result_dot;
    }
}

void ModelHawkesSumExpCustom::grad(const ArrayDouble &coeffs,
                                            ArrayDouble &out) {
    if (!weights_computed) compute_weights();
    out.fill(0);

    // This allows to run in a multithreaded environment the computation of each component
    parallel_run(get_n_threads(),
                 n_nodes,
                 &ModelHawkesSumExpCustom::grad_dim_i,
                 this,
                 coeffs,
                 out);
    out /= n_total_jumps;

    for (ulong k = 0; k != get_n_coeffs(); ++k)
        out[k] = -out[k];
}