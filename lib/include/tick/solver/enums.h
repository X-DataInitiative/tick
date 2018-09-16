#ifndef LIB_INCLUDE_TICK_SOLVER_ENUMS_H_
#define LIB_INCLUDE_TICK_SOLVER_ENUMS_H_

// License: BSD 3 clause

enum class SVRG_VarianceReductionMethod : uint16_t {
  Last = 1,
  Average = 2,
  Random = 3,
};
inline std::ostream &operator<<(std::ostream &s,
                                const SVRG_VarianceReductionMethod r) {
  typedef std::underlying_type<SVRG_VarianceReductionMethod>::type utype;
  return s << static_cast<utype>(r);
}

enum class SVRG_StepType : uint16_t {
  Fixed = 1,
  BarzilaiBorwein = 2,
};
inline std::ostream &operator<<(std::ostream &s, const SVRG_StepType r) {
  typedef std::underlying_type<SVRG_StepType>::type utype;
  return s << static_cast<utype>(r);
}

#endif  // LIB_INCLUDE_TICK_SOLVER_ENUMS_H_
