// License: BSD 3 clause

%module prox

%include defs.i
%include std_shared_ptr.i

%shared_ptr(TProx<double>);
%shared_ptr(TProx<float>);
%shared_ptr(Prox);

%shared_ptr(TProxWithGroups<double>);
%shared_ptr(TProxWithGroups<float>);
%shared_ptr(ProxWithGroups);

%shared_ptr(TProxSeparable<double>);
%shared_ptr(TProxSeparable<float>);
%shared_ptr(ProxSeparable);

%shared_ptr(TProxZero<double>);
%shared_ptr(TProxZero<float>);
%shared_ptr(ProxZero);

%shared_ptr(TProxPositive<double>);
%shared_ptr(TProxPositive<float>);
%shared_ptr(ProxPositive);

%shared_ptr(TProxL2Sq<double>);
%shared_ptr(TProxL2Sq<float>);
%shared_ptr(ProxL2Sq);

%shared_ptr(ProxL1);
%shared_ptr(ProxL1Double);
%shared_ptr(ProxL1Float);

%shared_ptr(TProxL1w<double>);
%shared_ptr(TProxL1w<float>);
%shared_ptr(ProxL1w);

%shared_ptr(TProxTV<double>);
%shared_ptr(TProxTV<float>);
%shared_ptr(ProxTV);

%shared_ptr(TProxElasticNet<double>);
%shared_ptr(TProxElasticNet<float>);
%shared_ptr(ProxElasticNet);

%shared_ptr(TProxSlope<double>);
%shared_ptr(TProxSlope<float>);
%shared_ptr(ProxSlope);

%shared_ptr(TProxMulti<double>);
%shared_ptr(TProxMulti<float>);
%shared_ptr(ProxMulti);

%shared_ptr(TProxEquality<double>);
%shared_ptr(TProxEquality<float>);
%shared_ptr(ProxEquality);

%shared_ptr(TProxBinarsity<double>);
%shared_ptr(TProxBinarsity<float>);
%shared_ptr(ProxBinarsity);

%shared_ptr(TProxL2<double>);
%shared_ptr(TProxL2<float>);
%shared_ptr(ProxL2);

%shared_ptr(TProxGroupL1<double>);
%shared_ptr(TProxGroupL1<float>);
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
