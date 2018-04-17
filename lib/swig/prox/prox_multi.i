// License: BSD 3 clause


%{
#include "tick/prox/prox_multi.h"
%}

%include "prox.i"

%template(ProxDoublePtrVector) std::vector<ProxDoublePtr>;
%template(ProxFloatPtrVector)  std::vector<ProxFloatPtr>;

template <class T>
class TProxMulti : public TProx<T> {
 public:
   explicit TProxMulti(std::vector<std::shared_ptr<TProx<T> > > proxs);

  bool compare(const TProxMulti<T> &that);
};

%rename(ProxMultiDouble) TProxMulti<double>;
class TProxMulti<double> : public TProx<double> {
 public:
  ProxMultiDouble(std::vector<std::shared_ptr<TProx<double> > > proxs);
};
typedef TProxMulti<double> ProxMultiDouble;

%rename(ProxMultiFloat) TProxMulti<float>;
class TProxMulti<float> : public TProx<float> {
 public:
  ProxMultiFloat(std::vector<std::shared_ptr<TProx<float> > > proxs);
};
typedef TProxMulti<double> ProxMultiDouble;
