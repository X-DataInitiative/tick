// License: BSD 3 clause

%{
#include "tick/optim/prox/prox_multi.h"
%}

%include "std_vector.i"
%template(ProxPtrVector) std::vector<ProxPtr>;

class ProxMulti : public Prox {
 public:
   ProxMulti(std::vector<ProxPtr> proxs);
};
