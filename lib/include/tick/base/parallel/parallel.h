#ifndef LIB_INCLUDE_TICK_BASE_PARALLEL_PARALLEL_H_
#define LIB_INCLUDE_TICK_BASE_PARALLEL_PARALLEL_H_

// License: BSD 3 clause

#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <type_traits>
#include <array>
#include <functional>

#include "tick/base/interruption.h"
#include "parallel_utils.h"

/*
 * This file implements templates for parallel computing of a method f(i,...) for a range of i.
 * There are mainly two templates :
 *      1. One which does not care about the returning value of f : parallel_run
 *      2. One which returns the returned values in an array : parallel_map
 *         In this case, there are two sub-cases :
 *         a- f(...) returns a type taken care by the SArray<V> (e.g., V=double,float,long,...)
 *            The collected returned values are stored in an SArray<V>Ptr :
 *                  SArray<V>Ptr parallel_map(...)
 *         b- f(...) returns a type not taken care by the SArray<V>Ptr
 *            The collected returned values are stored in an std::vector<V>
 *                  std::vector<V> parallel_map(...)
 */

namespace tick {

inline std::tuple<ulong, ulong> get_thread_indices(unsigned int thread_num, unsigned int num_threads, ulong dim) {
    if (dim < num_threads)
        return std::make_tuple(thread_num, thread_num + 1);

    return std::make_tuple(
        (thread_num * dim) / num_threads,
        std::min(((thread_num + 1) * dim) / num_threads, dim));
}

}  // namespace tick


/// @cond

// This is the function that will be called on each thread
// It execute a given function on the given params and store the result in
// map_result
// We used to execute this lambda function
// [&map_result, &next_i, dim, &f, &obj, &args...]() {
//     while (next_i < dim) {
//         ulong i = next_i.fetch_add(1);
//         if (i >= dim) return;
//         map_result[i] = (obj->*f)(i, args...);
//     }
// }
// But with old gcc (< 4.9) templates and lambda functions do not compile


template<typename R, typename T, typename S, typename... Args>
void _parallel_map_execute_task_and_store_result(R &map_result,
                                                 unsigned int thread_num,
                                                 unsigned int num_threads,
                                                 ulong dim,
                                                 T &f,
                                                 S &obj,
                                                 std::exception_ptr &ex,
                                                 Args &&... args) {
    ulong min_index{}, max_index{};

    std::tie(min_index, max_index) = tick::get_thread_indices(thread_num, num_threads, dim);

    try {
        for (ulong i = min_index; i < max_index; ++i) {
            map_result[i] = (obj->*f)(i, args...);
        }
    }
        // If an interruption was thrown we just return.
        // The Interruption flag is set and will be dealt during the join
    catch (...) {
        ex = std::current_exception();
    }
}

/*
 * This is a template to be called only by the templates below
 * It implements the parallelization in the cases 2. and 3.
 * collecting the result in a structure of type R (which is either Array<V> or
 * std::vector<V>)
 *
 * Look doc below to understand what the parameters are.
 */

template<typename R, typename T, typename S, typename... Args>
void _parallel_map(R &map_result,
                   unsigned int n_threads,
                   ulong dim,
                   T f,
                   S obj,
                   Args &&... args) {
    // if n_threads <= 1, we run the computation with no thread
    if (n_threads <= 1) {
        for (ulong i = 0; i < dim; i++)
            map_result[i] = (obj->*f)(i, args...);

        Interruption::throw_if_raised();
    } else {
        std::vector<std::thread> threads;
        std::vector<std::exception_ptr> exceptions{n_threads};

        for (unsigned int n = 0; n < std::min(static_cast<ulong>(n_threads), dim); n++) {
            threads.push_back(std::thread(
                _parallel_map_execute_task_and_store_result<R, T, S, Args...>,
                std::ref(map_result),
                n,
                n_threads,
                dim,
                std::ref(f),
                std::ref(obj),
                std::ref(exceptions[n]),
                std::ref(args)...));
        }

        for (auto &thread : threads) {
            thread.join();
        }

        tick::rethrow_exceptions(exceptions);

        // Throw an exception if interruption was detected while using a thread
        Interruption::throw_if_raised();
    }
}

/// @endcond

#include "tick/array/view.h"
/**
 * @brief This template allows to call, in a multi-threaded environment, a method of a class on
 * independent data referred to by an index. This template is used when the method returns a value
 * whose type V can be used in an SArray<V>Ptr class.
 *
 * \param n_threads : the number of threads to use (if 0 or 1 then everything is
 * sequential, no thread is used)
 *
 * \param dim : the number of independent data
 *
 * \param f : a pointer to the method to be called (the first argument of this
 * method should be the ulong referring to the index)
 *
 * \param obj : the object the method should be called on (generally 'this')
 *
 * \param args : the other arguments of f
 *
 * \return returns the collected return values in an SArray<V>Ptr object
 */
template<typename T, typename S, typename... Args>
auto parallel_map(unsigned int n_threads,
                  ulong dim,
                  T f,
                  S obj,
                  Args &&... args)
// This template is only used if f returns a non void type V for which we can
// build an SArray<V>Ptr class
// ==> integral and floating point types
-> typename tick::enable_if_python_primitive<T, S, Args...>::type {
    using return_type = tick::FuncResultType<T, S, Args...>;

    std::shared_ptr<SArray<return_type>> map_result = std::make_shared<SArray<return_type>>(dim);

    // Creating a view that will be filled during the parallel_map
    // One should avoid using SArrayDoublePtr in parallel_map since, the reference counting
    // part of this class is atomic which will result in slowing done the multithreading
    Array<return_type> view1 = view(*map_result);

    // We forward the call to the _parallel_map template
    _parallel_map<Array<return_type>>(view1, n_threads, dim, f, obj, args...);

    return map_result;
}

/**
 * @brief This template allows to call, in a multi-threaded environment, a method of a class on
 * independent data referred to by an index. This template is used when the method returns a value
 * whose type V cannot be used in an SArray<V>Ptr class. Thus the returned data are collected in
 * an std::vector<V> object.
 *
 * \param n_threads : the number of threads to use (if 0 or 1 then everything is
 * sequential, no thread is used)
 *
 * \param dim : the number of independent data
 *
 * \param f : a pointer to the method to be called (the first argument of this
 * method should be the ulong referring to the index)
 *
 * \param obj : the object the method should be called on (generally 'this')
 *
 * \param args : the other arguments of f
 *
 * \return returns the collected return values in an std::vector<V> object
 */
template<typename T, typename S, typename... Args>
auto parallel_map(unsigned int n_threads,
                  ulong dim,
                  T f,
                  S obj,
                  Args &&... args)
// This template is only used if f returns a non void type V which cannot be used to build an
// SArray(view,n_threads,dim,f,obj,args.<V>Ptr class
// ==> no integral nor floating_point types
-> typename tick::enable_if_not_python_primitive<T, S, Args...>::type {
    using return_type = tick::FuncResultType<T, S, Args...>;

    std::vector<return_type> map_result(dim);

    // We forward the call to the _parallel_map template
    _parallel_map<std::vector<return_type>>(map_result, n_threads, dim, f, obj, args...);

    return map_result;
}


// This is the function that will be called on each thread
// It execute a given function on the given params and store the result in
// map_result
// We used to execute this lambda function
// [&map_result, &next_i, dim, &f, &obj, &args...]() {
//     while (next_i < dim) {
//         ulong i = next_i.fetch_add(1);
//         if (i >= dim) return;
//         map_result[i] = (obj->*f)(i, args...);
//     }
// }
// But with old gcc (< 4.9) templates and lambda functions do not compile



/**
 * @brief This template allows to call, in a multi-threaded environment, a method of a class on
 * independent data referred to by an index. This template is used when one does not care about
 * the values returned by f.
 *
 * \param n_threads : the number of threads to use (if 0 or 1 then everything is
 * sequential, no thread is used)
 *
 * \param dim : the number of independent data
 *
 * \param f : a pointer to the method to be called (the first argument of this
 * method should be the ulong referring to the index)
 *
 * \param obj : the object the method should be called on (generally 'this')
 *
 * \param args : the other arguments of f
 *
 */


/// @cond

// This is the function that will be called on each thread
// It execute a given function on the given params and discards the result
// We used to execute this lambda function
// [&next_i, dim, &f, &obj, &args...]() {
//     while (next_i < dim) {
//         ulong i = next_i.fetch_add(1);
//         if (i>=dim) return;
//         (obj->*f)(i, args...);
//     }
// }
// But with old gcc (< 4.9) templates and lambda functions do not compile

template<typename T, typename S, typename... Args>
void _parallel_run_execute_task(
    unsigned int thread_num,
    unsigned int num_threads,
    ulong dim,
    T &f,
    S &obj,
    std::exception_ptr &ex,
    Args &&... args) {
    ulong min_index{}, max_index{};

    std::tie(min_index, max_index) = tick::get_thread_indices(thread_num, num_threads, dim);

    try {
        for (ulong i = min_index; i < max_index; ++i) {
            (obj->*f)(i, args...);
        }
    }
        // If an interruption was thrown we just return.
        // The Interruption flag is set and will be dealt during the join
    catch (...) {
        ex = std::current_exception();
    }
}

/// @endcond


template<typename T, typename S, typename... Args>
void parallel_run(unsigned int n_threads,
                  ulong dim,
                  T f,
                  S obj,
                  Args &&... args) {
    // if n_threads <= 1, we run the computation with no thread
    if (n_threads <= 1) {
        for (ulong i = 0; i < dim; i++)
            (obj->*f)(i, args...);

        Interruption::throw_if_raised();
    } else {
        std::vector<std::thread> threads;
        std::vector<std::exception_ptr> exceptions{n_threads};

        for (unsigned int n = 0; n < std::min(static_cast<ulong>(n_threads), dim); n++) {
            threads.push_back(std::thread(
                _parallel_run_execute_task<T, S, Args...>,
                n,
                n_threads,
                dim,
                std::ref(f),
                std::ref(obj),
                std::ref(exceptions[n]),
                std::ref(args)...));
        }

        for (auto &thread : threads) {
            thread.join();
        }

        tick::rethrow_exceptions(exceptions);

        // Throw an exception if interruption was detected while using a thread
        Interruption::throw_if_raised();
    }
}

/// @cond

// This is the function that will be called on each thread
// It execute a given function on the given params and call the reduce function to merge the result
// of current thread to previous result
// reduce function must take as first argument the previous result, as second argument, the result
// of thread i and return the result of the merged result
template<typename T, typename S, typename BinaryOp, typename... Args>
void _parallel_map_execute_task_and_reduce_result(unsigned int thread_num,
                                                  unsigned int num_threads,
                                                  ulong dim,
                                                  BinaryOp reduce_function,
                                                  T &f,
                                                  S &obj,
                                                  std::exception_ptr &ex,
                                                  typename tick::FuncResultType<T, S, Args...> &result_ref,
                                                  Args &&... args) {
    ulong min_index{}, max_index{};

    std::tie(min_index, max_index) = tick::get_thread_indices(thread_num, num_threads, dim);

    try {
        for (ulong i = min_index; i < max_index; ++i) {
            result_ref = reduce_function(result_ref, (obj->*f)(i, args...));
        }
    }
        // If an interruption was thrown we just return.
        // The Interruption flag is set and will be dealt during the join
    catch (...) {
        ex = std::current_exception();
    }
}
/// @endcond


/**
 * @brief This template allows to call, in a multi-threaded environment, a method of a class on
 * independent data referred to by an index. The the method must returns a value
 * which be given to a reduce function that will update the current result with it
 *
 * \param n_threads : the number of threads to use (if 0 or 1 then everything is
 * sequential, no thread is used)
 *
 * \param dim : the number of independent data
 *
 * \param reduce_function : a reduce function which takes as argument two value of
 * the type returned by f and reduce them into one result of the same type. For example, if we want
 * to return the sum of all our results, and ouf function f return double we can define it with:
 *
 *       double add(double left, double right){
 *           return left + right;
 *       }
 *
 * \param f : a pointer to the method to be called (the first argument of this
 * method should be the ulong referring to the index)
 *
 * \param obj : the object the method should be called on (generally 'this')
 *
 * \param args : the other arguments of f
 *
 * \return returns the reduced result
 */
template<typename T, typename S, typename BinaryOp, typename... Args>
auto parallel_map_reduce(unsigned int n_threads,
                         ulong dim,
                         BinaryOp reduce_function,
                         T f,
                         S obj,
                         Args &&... args)
-> typename tick::FuncResultType<T, S, Args...> {
    // RT stands for return type
    using RT =  typename tick::FuncResultType<T, S, Args...>;

    std::vector<RT> local_results(n_threads, RT{});

    // if n_threads == 1, we run the computation with no thread
    if (n_threads == 1) {
        for (ulong i = 0; i < dim; i++)
            local_results[0] = reduce_function(local_results[0], (obj->*f)(i, args...));

        Interruption::throw_if_raised();
    } else {
        std::vector<std::thread> threads;
        std::vector<std::exception_ptr> exceptions{n_threads};

        for (unsigned int n = 0; n < std::min(static_cast<ulong>(n_threads), dim); n++) {
            threads.push_back(std::thread(
                _parallel_map_execute_task_and_reduce_result<T, S, BinaryOp, Args...>,
                n,
                n_threads,
                dim,
                reduce_function,
                std::ref(f),
                std::ref(obj),
                std::ref(exceptions[n]),
                std::ref(local_results[n]),
                std::ref(args)...));
        }

        for (auto &thread : threads) {
            thread.join();
        }

        tick::rethrow_exceptions(exceptions);

        Interruption::throw_if_raised();
    }

    {
        RT result{};

        for (auto &r : local_results) {
            result = reduce_function(result, r);
        }

        return result;
    }
}

/// @cond

/// @endcond


template<typename R, typename Functor, typename... Args>
void _parallel_map_array_execute_task_and_reduce_result(unsigned int thread_num,
                                                        unsigned int num_threads,
                                                        ulong dim,
                                                        Functor f,
                                                        R &local_result,
                                                        std::exception_ptr &ex,
                                                        Args &... args) {
    ulong min_index{}, max_index{};

    std::tie(min_index, max_index) = tick::get_thread_indices(thread_num, num_threads, dim);

    try {
        for (ulong i = min_index; i < max_index; ++i) {
            f(i, local_result, args...);
        }
    } catch (...) {
        // If an interruption was thrown we just return.
        // The Interruption flag is set and will be dealt during the join

        ex = std::current_exception();
    }
}

/**
 * @brief Reduction of arrays into arrays
 *
 * This parallel function is similar to the 'parallel_map_reduce', with the exception that the current thread-local
 * result (state) is given by reference to the called functor 'f' as the first argument. The functor must update the
 * state instead of returning a result.
 *
 * Also, the reduction function must update the first/left-most reference parameter instead of returning a value.
 *
 * @param n_threads Number of threads to execute for this parallel task
 * @param dim Number of tasks. Tasks are split into even groups and assigned to each thread
 * @param redux Reduction function. Must take the form 'void(T& state, const U& item)'
 * @param f Functor object. Must take the form 'void(ulong idx, T& state, Args...& args)'
 * @param out Output reference. Also used to initialize the thread-local results
 * @param args Custom arguments passed to the functor
 */
template<typename R, typename Functor, typename BinaryOp, typename... Args>
void parallel_map_array(unsigned int n_threads,
                        ulong dim,
                        BinaryOp redux,
                        Functor f,
                        R &out,
                        Args &... args) {
    std::vector<R> local_results(n_threads, out);

    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> exceptions{n_threads};

    for (unsigned int n = 0; n < std::min(static_cast<ulong>(n_threads), dim); n++) {
        threads.push_back(std::thread(
            _parallel_map_array_execute_task_and_reduce_result<R, Functor, Args...>,
            n,
            n_threads,
            dim,
            std::ref(f),
            std::ref(local_results[n]),
            std::ref(exceptions[n]),
            std::ref(args)...));
    }

    for (auto &thread : threads) {
        thread.join();
    }

    for (auto &local_result : local_results) {
        redux(out, local_result);
    }
}

/**
 * Identical to the other parallel_map_array, except this version takes a member function pointer plus an object instead
 * of a functor.
 *
 * @param n_threads Number of threads to execute for this parallel task
 * @param dim Number of tasks. Tasks are split into even groups and assigned to each thread
 * @param redux Reduction function. Must take the form 'void(T& state, const U& item)'
 * @param f Member function pointer to be called for each index value.
 * @param obj Object on which to invoke the member function pointer
 * @param out Output reference. Also used to initialize the thread-local results
 * @param args Custom arguments passed to the functor
 */
template<typename R, typename T, typename S, typename BinaryOp, typename... Args>
void parallel_map_array(unsigned int n_threads,
                        ulong dim,
                        BinaryOp redux,
                        T f,
                        S* obj,
                        R &out,
                        Args &... args) {
    using std::placeholders::_1;
    using std::placeholders::_2;

    parallel_map_array<R>(n_threads, dim, redux, std::bind(f, obj, _1, _2, std::ref(args)...), out);
}

/**
 * @brief This template allows to call, in a multi-threaded environment, a method of a class on
 * independent data referred to by an index. Each time the method returns a result, it will be
 * added to the sum of the previous ones
 *
 * \param n_threads : the number of threads to use (if 0 or 1 then everything is
 * sequential, no thread is used)
 *
 * \param dim : the number of independent data

 * \param f : a pointer to the method to be called (the first argument of this
 * method should be the ulong referring to the index)
 *
 * \param obj : the object the method should be called on (generally 'this')
 *
 * \param args : the other arguments of f
 *
 * \return returns the sum of all results
 */

template<typename T, typename S, typename... Args>
auto parallel_map_additive_reduce(unsigned int n_threads,
                                  ulong dim,
                                  T f,
                                  S obj,
                                  Args &&... args)
-> typename tick::FuncResultType<T, S, Args...> {
    using RT =  typename tick::FuncResultType<T, S, Args...>;

    return parallel_map_reduce(n_threads, dim, std::plus<RT>{}, f, obj, args...);
};

#endif  // LIB_INCLUDE_TICK_BASE_PARALLEL_PARALLEL_H_
