// License: BSD 3 clause

#include "tick/prox/prox_binarsity.h"

template class DLL_PUBLIC TProxBinarsity<double, double>;
template class DLL_PUBLIC TProxBinarsity<float, float>;

template class DLL_PUBLIC TProxBinarsity<double, std::atomic<double>>;
template class DLL_PUBLIC TProxBinarsity<float, std::atomic<float>>;
