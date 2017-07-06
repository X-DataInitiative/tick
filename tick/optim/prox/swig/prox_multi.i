// License: BSD 3 clause

%{
#include "prox_multi.h"
%}

%include "std_vector.i"
%template(ProxPtrVector) std::vector<ProxPtr>;

class ProxMulti : public Prox {
 public:
   ProxMulti(std::vector<ProxPtr> proxs);
};
