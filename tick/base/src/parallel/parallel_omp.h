#ifndef TICK_BASE_SRC_PARALLEL_PARALLEL_OMP_H_
#define TICK_BASE_SRC_PARALLEL_PARALLEL_OMP_H_

#include <vector>
#include <thread>
#include <exception>
#include <iostream>
#include <type_traits>

#include <omp.h>

#include "interruption.h"
#include "parallel_utils.h"

/**
 * Performs a number 'dim' function calls of method 'f' on object 'obj' with arguments 'args' on a number 'n_threads' of
 * threads.
 *
 * If any of the function calls results in an exception or SIGINT, an exception will be thrown.
 *
 * @param n_threads Number of parallel threads
 * @param dim Number of function calls
 * @param f Method pointer. Signature must be S::T(ulong, Args...), i.e. first parameter must be an unsigned
 * long, which is the index (starting from 0) of the particular functional call
 * @param obj Object on which to invoke the method pointer
 * @param args Additional arguments to the method call
 */
template<typename T, typename S, typename... Args>
void parallel_run(unsigned int n_threads,
                  ulong dim,
                  T f,
                  S obj,
                  Args &&... args) {
  std::vector<std::exception_ptr> exceptions{n_threads};

#pragma omp parallel num_threads(n_threads)
  {
    bool has_thrown{false};

#pragma omp for schedule(static)
    for (ulong i = 0; i < dim; ++i) {
      try {
        if (!has_thrown)
          (obj->*f)(i, args...);
      } catch (...) {
        exceptions[omp_get_thread_num()] = std::current_exception();

        has_thrown = true;
      }
    }
  }

  tick::rethrow_exceptions(exceptions);

  Interruption::throw_if_raised();
}

namespace detail {

template<typename R, typename T, typename S, typename... Args>
void parallel_map_data(R *map_result,
                       unsigned int n_threads,
                       ulong dim,
                       T f,
                       S obj,
                       Args &&... args) {
  bool has_thrown{false};
  std::vector<std::exception_ptr> exceptions{n_threads};

  #pragma omp parallel for num_threads(n_threads) firstprivate(has_thrown)
  for (ulong i = 0; i < dim; ++i) {
    try {
      if (!has_thrown)
        map_result[i] = (obj->*f)(i, args...);
    } catch (std::exception &e) {
      exceptions[omp_get_thread_num()] = std::current_exception();

      has_thrown = true;
    }
  }

  tick::rethrow_exceptions(exceptions);

  Interruption::throw_if_raised();
}

}

template<typename T, typename S, typename... Args>
auto parallel_map(unsigned int n_threads,
                  ulong dim,
                  T f,
                  S obj,
                  Args &&... args)
-> typename tick::enable_if_python_primitive<T, S, Args...>::type {
  using return_type = tick::FuncResultType<T, S, Args...>;

  std::shared_ptr<SArray<return_type>> map_result = std::make_shared<SArray<return_type>>(dim);

  detail::parallel_map_data<return_type, T, S, Args...>(map_result->data(), n_threads, dim, f, obj, args...);

  return map_result;
};

template<typename T, typename S, typename... Args>
auto parallel_map(unsigned int n_threads,
                  ulong dim,
                  T f,
                  S obj,
                  Args &&... args)
-> typename tick::enable_if_not_python_primitive<T, S, Args...>::type {
  using return_type = tick::FuncResultType<T, S, Args...>;

  std::vector<return_type> map_result(dim);

  detail::parallel_map_data<return_type, T, S, Args...>(map_result.data(), n_threads, dim, f, obj, args...);

  return map_result;
};

template<typename T, typename S, typename U, typename... Args>
auto parallel_map_reduce(unsigned int n_threads,
                         ulong dim,
                         U reduce_function,
                         T f,
                         S obj,
                         Args &&... args)
-> tick::FuncResultType<T, S, Args...> {
  using RT = tick::FuncResultType<T, S, Args...>;

  bool has_thrown{false};
  std::vector<std::exception_ptr> exceptions{n_threads};
  std::vector<RT> local_results(n_threads, RT{});

  #pragma omp parallel num_threads(n_threads) firstprivate(has_thrown)
  {
    auto &local_result = local_results[omp_get_thread_num()];

    #pragma omp for
    for (ulong i = 0; i < dim; ++i) {
      try {
        if (!has_thrown)
          local_result = (*reduce_function)(local_result, (obj->*f)(i, args...));
      } catch (std::exception &e) {
        exceptions[omp_get_thread_num()] = std::current_exception();

        has_thrown = true;
      }
    }
  }

  tick::rethrow_exceptions(exceptions);

  {
    RT result{};

    for (auto &r : local_results) {
      result = (*reduce_function)(result, r);
    }

    return result;
  }
};

template<typename T, typename S, typename... Args>
auto parallel_map_additive_reduce(unsigned int n_threads,
                                  ulong dim,
                                  T f,
                                  S obj,
                                  Args &&... args)
-> tick::FuncResultType<T, S, Args...> {
  using RT = tick::FuncResultType<T, S, Args...>;

  RT sum{};

  bool has_thrown{false};
  std::vector<std::exception_ptr> exceptions{n_threads};

  #pragma omp parallel for num_threads(n_threads) firstprivate(has_thrown) reduction(+:sum)
  for (ulong i = 0; i < dim; ++i) {
    try {
      if (!has_thrown)
        sum += (obj->*f)(i, args...);
    } catch (std::exception &e) {
      exceptions[omp_get_thread_num()] = std::current_exception();

      has_thrown = true;
    }
  }

  tick::rethrow_exceptions(exceptions);

  return sum;
};

template<typename R, typename Functor, typename BinaryOp, typename... Args>
void parallel_map_array(unsigned int n_threads,
                        ulong dim,
                        BinaryOp redux,
                        Functor f,
                        R &out,
                        Args &... args) {
  std::vector<R> local_results(n_threads, out);

  bool has_thrown{false};
  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> exceptions{n_threads};

  #pragma omp parallel for num_threads(n_threads) firstprivate(has_thrown)
  for (ulong i = 0; i < dim; ++i) {
    try {
      if (!has_thrown)
        f(i, local_results[omp_get_thread_num()], args...);
    } catch (std::exception &e) {
      exceptions[omp_get_thread_num()] = std::current_exception();

      has_thrown = true;
    }
  }

  tick::rethrow_exceptions(exceptions);

  for (auto &local_result : local_results) {
    redux(out, local_result);
  }
}

#endif  // TICK_BASE_SRC_PARALLEL_PARALLEL_OMP_H_
