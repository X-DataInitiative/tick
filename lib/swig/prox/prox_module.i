// License: BSD 3 clause

%module prox

%include defs.i
%include std_shared_ptr.i

%shared_ptr(Prox);
%shared_ptr(TProx<double, double>);
%shared_ptr(TProx<float, float>);
%shared_ptr(TProx<double, std::atomic<double>>);
%shared_ptr(TProx<float, std::atomic<float>>);

%shared_ptr(TProxWithGroups<double, double>);
%shared_ptr(TProxWithGroups<float, float>);
%shared_ptr(ProxWithGroups);

%shared_ptr(TProxSeparable<double, double>);
%shared_ptr(TProxSeparable<float, float>);
%shared_ptr(TProxSeparable<double, std::atomic<double>>);
%shared_ptr(TProxSeparable<float, std::atomic<float>>);
%shared_ptr(ProxSeparable);

%shared_ptr(TProxZero<double, double>);
%shared_ptr(TProxZero<float, float>);
%shared_ptr(ProxZero);

%shared_ptr(TProxPositive<double, double>);
%shared_ptr(TProxPositive<float, float>);
%shared_ptr(ProxPositive);

%shared_ptr(TProxL2Sq<double, double>);
%shared_ptr(TProxL2Sq<float, float>);
%shared_ptr(ProxL2Sq);

%shared_ptr(ProxL1);
%shared_ptr(ProxL1Double);
%shared_ptr(ProxL1Float);

%shared_ptr(TProxL1w<double, double>);
%shared_ptr(TProxL1w<float, float>);
%shared_ptr(ProxL1w);

%shared_ptr(TProxTV<double, double>);
%shared_ptr(TProxTV<float, float>);
%shared_ptr(ProxTV);

%shared_ptr(TProxElasticNet<double, double>);
%shared_ptr(TProxElasticNet<float, float>);
%shared_ptr(TProxElasticNet<double, std::atomic<double>>);
%shared_ptr(TProxElasticNet<float, std::atomic<float>>);
%shared_ptr(ProxElasticNet);

%shared_ptr(TProxSlope<double, double>);
%shared_ptr(TProxSlope<float, float>);
%shared_ptr(ProxSlope);

%shared_ptr(TProxMulti<double, double>);
%shared_ptr(TProxMulti<float, float>);
%shared_ptr(ProxMulti);

%shared_ptr(TProxEquality<double, double>);
%shared_ptr(TProxEquality<float, float>);
%shared_ptr(ProxEquality);

%shared_ptr(TProxBinarsity<double, double>);
%shared_ptr(TProxBinarsity<float, float>);
%shared_ptr(ProxBinarsity);

%shared_ptr(TProxL2<double, double>);
%shared_ptr(TProxL2<float, float>);
%shared_ptr(ProxL2);

%shared_ptr(TProxGroupL1<double, double>);
%shared_ptr(TProxGroupL1<float, float>);
%shared_ptr(ProxGroupL1);

%{
#include "tick/base/tick_python.h"
#include "tick/base/serialization.h"
%}

%import(module="tick.base") base_module.i

%include serialization.i

%include "prox.i"

%include "prox_separable.i"

%include prox_zero.i

%include prox_positive.i

%include prox_l2sq.i

%include prox_l2.i

%include prox_l1.i

%include prox_l1w.i

%include prox_tv.i

%include prox_elasticnet.i

%include prox_slope.i

%include prox_multi.i

%include prox_equality.i

%include prox_binarsity.i

%include prox_group_l1.i
