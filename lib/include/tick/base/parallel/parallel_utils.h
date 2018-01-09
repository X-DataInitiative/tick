//
// Created by poulsen on 10/24/16.
//

#ifndef LIB_INCLUDE_TICK_BASE_PARALLEL_PARALLEL_UTILS_H_
#define LIB_INCLUDE_TICK_BASE_PARALLEL_PARALLEL_UTILS_H_

// License: BSD 3 clause

#include <type_traits>
#include "tick/array/sarray.h"

namespace tick {

/**
 * Determines the return value resulting from calling method type 'T' on object type 'S' with parameter types
 * 'Args...'
 */
template<typename T, typename S, typename... Args>
using FuncResultType = typename std::result_of<T(S, ulong, Args...)>::type;

/**
 * Determine if return type of function call is a Python primitive or not
 *
 * @return True if return type of function call is a Python primitive, otherwise false
 */
template<typename T, typename S, typename... Args>
constexpr bool result_is_python_primitive() {
    using RT = FuncResultType<T, S, Args...>;

    return std::is_integral<RT>::value || std::is_floating_point<RT>::value;
}

template<typename T, typename S, typename... Args>
using enable_if_python_primitive = typename std::enable_if<result_is_python_primitive<T, S, Args...>(),
                                                           std::shared_ptr<SArray<FuncResultType<T, S, Args...>>>>;

template<typename T, typename S, typename... Args>
using enable_if_not_python_primitive = typename std::enable_if<!result_is_python_primitive<T, S, Args...>(),
                                                               std::vector<FuncResultType<T, S, Args...>>>;

/**
 * Utility function to rethrow the first of a collection of possible exceptions.
 *
 * Will not throw if all exception pointers a nullpointers.
 *
 * @param exceptions Collection of possible exceptions (represented by set of std::exception_ptr)
 */
inline void rethrow_exceptions(std::vector<std::exception_ptr> &exceptions) {
    for (auto &eptr : exceptions) {
        if (eptr != nullptr) {
            std::rethrow_exception(eptr);
        }
    }
}

/**
 * Primary template for the return type traits for parallel_map.
 *
 * It is specialized in two cases, one for return types we can return to Python (integers and floating points) and
 * one for all other types.
 *
 * Each specialization defines 'type' which is the returned collection, and 'RT' which is the value type of the
 * returned collection.
 */
template<typename T, typename S, class Enable = void, typename... Args>
struct map_return_t {};

/**
 * Specialization for primitive types (integers and floating points).
 *
 * Type returned is a shared SArray.
 */
template<typename T, typename S, typename... Args>
struct map_return_t<T, S, typename enable_if_python_primitive<T, S, Args...>::type, Args...> {
    using RT = tick::FuncResultType<T, S, Args...>;
    using type = typename enable_if_python_primitive<T, S, Args...>::type;

    type create(ulong dim) const { return std::make_shared<SArray<RT>>(dim); }
    RT *data(type &t) const { return t->data(); }
};

/**
 * Specialization for non-primitive types.
 *
 * Type returned is a std::vector.
 */
template<typename T, typename S, typename... Args>
struct map_return_t<T, S, typename enable_if_not_python_primitive<T, S, Args...>::type, Args...> {
    using RT = tick::FuncResultType<T, S, Args...>;
    using type = std::vector<RT>;

    type create(ulong dim) const { return std::vector<RT>(dim); }
    RT *data(type &t) const { return t.data(); }
};
}  // namespace tick

#endif  // LIB_INCLUDE_TICK_BASE_PARALLEL_PARALLEL_UTILS_H_
