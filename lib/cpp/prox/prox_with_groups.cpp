// License: BSD 3 clause

#include "tick/prox/prox_with_groups.h"

template class TProxWithGroups<double, double>;
template class TProxWithGroups<float, float>;

template class TProxWithGroups<double, std::atomic<double>>;
template class TProxWithGroups<float, std::atomic<float>>;
