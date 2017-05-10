%{
#include "prox_slope.h"
%}


enum class WeightsType {
    bh = 0,
    oscar
};


class ProxSlope : public Prox {

public:

    ProxSlope(double lambda, double fdr, bool positive);

    ProxSlope(double lambda, double fdr,
              unsigned long start, unsigned long end,
              bool positive);

    inline virtual double get_fdr() const;
    inline virtual void set_fdr(double fdr);

    inline virtual bool get_positive() const;
    inline virtual void set_positive(bool positive);

    inline double get_weight_i(unsigned long i);

    inline virtual double get_strength() const;
};
