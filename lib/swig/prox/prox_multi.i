// License: BSD 3 clause


%{
#include "tick/prox/prox_multi.h"
%}

%include "prox.i"

%template(ProxDoublePtrVector) std::vector<ProxDoublePtr>;

template <class T>
class TProxMulti : public TProx<T> {
 public:
   explicit TProxMulti(std::vector<std::shared_ptr<TProx<T> > > proxs);

  bool compare(const TProxMulti<T> &that);
};

%template(ProxMulti) TProxMulti<double>;
typedef TProxMulti<double> ProxMulti;

%template(ProxMultiDouble) TProxMulti<double>;
typedef TProxMulti<double> ProxMultiDouble;

%template(ProxMultiFloat) TProxMulti<float>;
typedef TProxMulti<float> ProxMultiFloat;
