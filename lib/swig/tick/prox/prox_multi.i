// License: BSD 3 clause


%{
#include "tick/prox/prox_multi.h"
%}

%include "prox.i"

%template(ProxDoublePtrVector) std::vector<ProxDoublePtr>;
%template(ProxFloatPtrVector)  std::vector<ProxFloatPtr>;

template <class T, class K>
class TProxMulti : public TProx<T, K> {
 public:
   explicit TProxMulti(std::vector<std::shared_ptr<TProx<T, K> > > proxs);

  bool compare(const TProxMulti<T, K> &that);
};

%rename(ProxMultiDouble) TProxMulti<double, double>;
class TProxMulti<double, double> : public TProx<double, double> {
 public:
  ProxMultiDouble(std::vector<std::shared_ptr<TProx<double, double> > > proxs);
};
typedef TProxMulti<double, double> ProxMultiDouble;

%rename(ProxMultiFloat) TProxMulti<float, float>;
class TProxMulti<float, float> : public TProx<float, float> {
 public:
  ProxMultiFloat(std::vector<std::shared_ptr<TProx<float, float> > > proxs);
};
typedef TProxMulti<float, float> ProxMultiFloat;

