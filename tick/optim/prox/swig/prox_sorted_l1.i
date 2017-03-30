%{
#include "prox_sorted_l1.h"
%}


enum class WeightsType {
    bh = 0,
    oscar
};


class ProxSortedL1 : public Prox {

public:

    ProxSortedL1(double lambda, double fdr, WeightsType weights_type,
                 bool positive);

    ProxSortedL1(double lambda, double fdr, WeightsType weights_type,
                 unsigned long start, unsigned long end,
                 bool positive);

    inline virtual double get_fdr() const;
    inline virtual void set_fdr(double fdr);

    inline virtual WeightsType get_weights_type() const;
    inline virtual void set_weights_type(WeightsType weights_type);

    inline virtual bool get_positive() const;
    inline virtual void set_positive(bool positive);

    inline double get_weight_i(unsigned long i);

};
