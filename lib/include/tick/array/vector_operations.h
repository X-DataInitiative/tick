
#ifndef LIB_INCLUDE_TICK_ARRAY_VECTOR_OPERATIONS_H_
#define LIB_INCLUDE_TICK_ARRAY_VECTOR_OPERATIONS_H_

// License: BSD 3 clause

#include <atomic>
#include <vector>
#include <numeric>
#include <algorithm>
#include <type_traits>

#include "promote.h"
#include "tick/base/defs.h"

#if defined(TICK_USE_MKL)

#include "mkl.h"

#elif defined(TICK_USE_CBLAS)

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
// TODO(svp) Disabling this feature until we find
//  a good way to determine if ATLAS is actually available
#else
extern "C" {
#include <cblas.h>
}
#endif  // defined(__APPLE__)

#else

#include "tick/array/vector/ops_unoptimized.h"
namespace tick {
template <typename T>
using vector_operations = detail::vector_operations_unoptimized<T>;
  }

#endif

#if defined(TICK_USE_MKL) || defined(TICK_USE_CBLAS)
#include "tick/array/vector/ops_blas.h"
namespace tick {
template <typename T>
using vector_operations = detail::vector_operations_cblas<T>;
  }
#endif

#include "tick/array/vector/ops_unoptimized_impl.h"

#endif  // LIB_INCLUDE_TICK_ARRAY_VECTOR_OPERATIONS_H_

