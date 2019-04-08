#ifndef LIB_INCLUDE_TICK_BASE_DEBUG_H_
#define LIB_INCLUDE_TICK_BASE_DEBUG_H_

// License: BSD 3 clause

/** @file
 *
 * Debugging utilities for 'tick'
 *
 * Available debug flags
 * (if you use the Python interface, these should be set in the setup.py file)

 * - DEBUG_C_ARRAY       : count #allocations of C-arrays
 * - DEBUG_ARRAY         : Track creation/destruction of Array objects
 * - DEBUG_SHAREDARRAY   : Track creation/destruction of SharedArray objects
 * - DEBUG_VARRAY        : Track VArray
 * - DEBUG_COSTLY_THROW  : Enables some costly tests to throw error
 *                         (such as Array[i] if i not in range)
 * - DEBUG_VERBOSE       : Error messages from CPP extensions will include
 *                         backtrace and error loc
*/

#include "defs.h"

#ifndef _WIN32

#include <execinfo.h>
#include <unistd.h>

#endif  // _WIN32

#include <string>
#include <iostream>
#include <sstream>
#include <exception>
#include <array>
#include <type_traits>

#ifdef DEBUG_VERBOSE
  #define TICK_DEBUG_VERBOSE_MODE 1
#endif

namespace tick {

/**
 * Utility class used to insert into a std::stringstream. On exit it calls the functor given as template parameter.
 *
 * @tparam ExitPolicy Functor called in deconstructor. Must take a std::string as first and only argument.
 */
template<typename ExitPolicy>
class TemporaryLog {
 private:
  std::stringstream ss;

 public:
  TemporaryLog() : ss() {}

  ~TemporaryLog() {
    ExitPolicy{}(ss.str());
  }

  TemporaryLog &insert_backtrace() {
#ifndef _WIN32
    std::array<void *, 100> stack_addresses;

    const int num_addresses = backtrace(stack_addresses.data(), stack_addresses.size());
    char **const strings = backtrace_symbols(stack_addresses.data(), num_addresses);

    (*this) << "C++ extension backtrace: \n";

    for (int j = 0; j < std::min(num_addresses, 10); ++j)
      (*this) << strings[j] << '\n';
#endif
    return *this;
  }

  std::stringstream& stream() { return ss; }

  std::string value() const { return ss.str(); }
};

template<typename E, typename T>
TemporaryLog<E>& operator<<(TemporaryLog<E>& log, const T &item) {
  log.stream() << item;

  return log;
}

template<typename E, typename T>
TemporaryLog<E>& operator<<(TemporaryLog<E>&& log, const T &item) {
  log.stream() << item;

  return log;
}

struct LogExitNoop {
  inline void operator()(const std::string &s) {}
};

struct LogExitCerr {
  inline void operator()(const std::string &s) { std::cerr << s << std::endl; }
};

struct LogExitCout {
  inline void operator()(const std::string &s) { std::cout << s << std::endl; }
};

}  // namespace tick

/**
 * Inserts filename, linenumber and function name into stream
 */
#ifdef _WIN32
#define TICK_LOG_PREFIX __FILE__ ":"  << __LINE__ << " in " << __FUNCTION__ << ": "
#else
#define TICK_LOG_PREFIX __FILE__ ":"  << __LINE__ << " in " << __PRETTY_FUNCTION__ << ": "
#endif

/**
 * \defgroup error_mod Error management
 * \brief Macros that can be used to throw errors that will be caught in Python
 * @{
 */

// If we're in debug mode a more explicit error exception is thrown, otherwise just a simple text
#ifdef TICK_DEBUG_VERBOSE_MODE

/**
  Macro for fatal errors. Throws a std::runtime_error.

  DEBUG version. Also prints backtrace and error loc.

  Example:
  \code{.cpp}
  TICK_ERROR("A fatal error occurred: " << 42);
  \endcode

  Anything that can be inserted into std::stringstream can also be inserted as a parameter here.
 */
  #define TICK_ERROR(fmt) \
  {throw std::runtime_error(((tick::TemporaryLog<tick::LogExitNoop>{} << TICK_LOG_PREFIX << fmt << '\n').insert_backtrace()).value());}

// If TICK_DEBUG_VERBOSE_MODE flag is not enabled, we throw a simpler error exception
#else  // TICK_DEBUG_VERBOSE_MODE

/**
  Macro for fatal errors. Throws a std::runtime_error.

  Example:
  \code{.cpp}
   TICK_ERROR("A fatal error occurred: " << 42);
  \endcode

  Anything that can be inserted into std::stringstream can also be inserted as a parameter here.
 */
#define TICK_ERROR(fmt) \
  {throw std::runtime_error(((tick::TemporaryLog<tick::LogExitNoop>{} << fmt << '\n')).value());}

#endif  // TICK_DEBUG_VERBOSE_MODE

/**
 * Use this macro to throw an error corresponding to a bad index use
 *
 * \param imin : the lowest available index
 * \param imax : the highest available index
 * \param i    : the requested index
 */
#define TICK_BAD_INDEX(imin, imax, i) \
  {throw std::out_of_range(((tick::TemporaryLog<tick::LogExitNoop>{} << TICK_LOG_PREFIX << "Bad index: " << i << " should be in [" << imin << ", " << imax << "]\n").insert_backtrace()).value());}

/**
  Macro for non-fatal user warnings printed to std::cerr.

  Example:
  \code{.cpp}
  TICK_WARNING() << "User has made a non-fatal mistake with number: " << 2;
  \endcode

  Anything that can be inserted into std::stringstream can also be inserted here.
 */
#define TICK_WARNING()  (tick::TemporaryLog<tick::LogExitCerr>{} << TICK_LOG_PREFIX)

/**
  Macro for non-fatal developer debug messages printed to std::cout.

  Should not make it into released code.

  Example:
  @code{.cpp}
  int* ptr = f();
  TICK_DEBUG() << "Debugging this pointer: " << ptr;
  @endcode

  Anything that can be inserted into std::stringstream can also be inserted here.
 */
#define TICK_DEBUG()    (tick::TemporaryLog<tick::LogExitCout>{} << TICK_LOG_PREFIX)

/**
 * Convenience macro to error the program if the current function is not implemented in class 'cls'
 */
#define TICK_CLASS_DOES_NOT_IMPLEMENT(cls) TICK_ERROR("Function not implemented in  " << cls)

/**
 * Macro to print current function stack.
 */
#define TICK_BACKTRACE() TICK_DEBUG().insert_backtrace()

/**@}*/

/**
  Convenience macro to define that method/member/variable/class/struct or typedef is deprecated.

  Will generate a warning on compilation if deprecated definition is used.

  Older versions of GCC omits the _reason_ text in the generated warning.

  Examples:
  \code{.cpp}
  TICK_DEPRECATED("Deprecated because of...")
  int some_method() {
    return int{};
  }

  TICK_DEPRECATED("Deprecated because of...")
  int my_variable;

  class TICK_DEPRECATED("This class has been replaced by...") my_class {
  };

  struct my_struct {
    TICK_DEPRECATED("Use other method instead")
    int struct_method() { return int{}; }
  };
  \endcode
 */
#if defined(__cpp_attributes)
  #define TICK_DEPRECATED(reason) [[deprecated(reason)]]
#elif defined(__GNUC__)
  #define TICK_DEPRECATED(reason) __attribute__ ((deprecated))
#elif defined(__clang__)
  #define TICK_DEPRECATED(reason) __attribute__((deprecated(reason)))
#else
  #define TICK_DEPRECATED(reason)
#endif

#endif  // LIB_INCLUDE_TICK_BASE_DEBUG_H_
