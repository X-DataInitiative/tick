// License: BSD 3 clause


%{
#include "tick/prox/prox_multi.h"
%}

%include "prox.i"

%template(ProxDoublePtrVector) std::vector<ProxDoublePtr>;

class ProxMulti : public ProxDouble {
 public:
   explicit ProxMulti(
     std::vector<std::shared_ptr<TProx<double, double> > > proxs
   );
};
