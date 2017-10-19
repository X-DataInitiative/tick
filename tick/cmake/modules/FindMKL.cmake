
set(INTEL_ROOT "/opt/intel")
set(INTEL_LIB "${INTEL_ROOT}/lib")
set(MKL_INCLUDE_DIRECTORIES "${INTEL_ROOT}/mkl/include")
set(MKL_LIB "${INTEL_ROOT}/mkl/lib")

file(GLOB MKL_LIBRARIES "${MKL_LIB}/*.dylib")
file(GLOB INTEL_LIBRARIES "${INTEL_LIB}/*.dylib")

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MKL "Did not found" INTEL_LIB MKL_INCLUDE_DIRECTORIES MKL_LIB)