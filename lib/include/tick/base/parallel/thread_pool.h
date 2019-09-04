#ifndef LIB_INCLUDE_TICK_BASE_PARALLEL_THREAD_POOL_H_
#define LIB_INCLUDE_TICK_BASE_PARALLEL_THREAD_POOL_H_

#include <mutex>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <condition_variable>
#include <unordered_map>
#include <iostream>

#include <future>
#include <chrono>

#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE > 200112L
#include <pthread.h>
#endif

namespace tick {

class ThreadPool{
 private:
  bool stop = 0;
  uint16_t m_n_threads;
  std::mutex m_mutex;
  std::condition_variable m_condition;
  std::atomic<uint16_t> m_done;
  std::vector<std::function<void()> > m_tasks;
  std::vector<std::thread> m_threads;
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE > 200112L
  uint16_t m_smt = 0;
  std::unordered_map<uint16_t, uint16_t> threadIX2NAlloc;
#endif

 public:
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool(const ThreadPool&&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&&) = delete;

  explicit ThreadPool(uint16_t n_threads, uint16_t smt = 0)
     : m_n_threads(n_threads), m_done(0)
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE > 200112L
     , m_smt(smt)
#endif
     {
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE > 200112L
    // Create a cpu_set_t object representing a set of CPUs. Clear it and mark
    auto num_cpus = std::thread::hardware_concurrency();
    cpu_set_t cpuset;
    for (uint16_t i = 0; i < m_n_threads; ++i) {
      threadIX2NAlloc[i] = 0;
    }
#endif

    for (uint16_t i = 0; i < m_n_threads; ++i) {
      m_threads.emplace_back([this](uint16_t ix) {
          for (;;) {
            std::function<void()> task;
            {
              std::unique_lock<std::mutex> lock(this->m_mutex);
              this->m_condition.wait(lock,
                  [this, ix]{ return this->m_tasks.size() > ix && m_done > ix; });
              if (this->stop) return;
              task = std::move(this->m_tasks[0]);
              this->m_tasks.erase(this->m_tasks.begin());
            }
            task();
            this->m_done--;
          }
        }, i);
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE > 200112L
      // only CPU i as set.
      CPU_ZERO(&cpuset);
      auto ix = num_cpus - (i + 1);
      if (m_smt) {
        ix = num_cpus - 1;
        while (threadIX2NAlloc[ix] == smt) {
          ix--;
        }
        threadIX2NAlloc[ix] += 1;
      }

      CPU_SET(ix, &cpuset);
      int rc = pthread_setaffinity_np(m_threads[i].native_handle(),
                                      sizeof(cpu_set_t), &cpuset);
      if (rc != 0) {
        std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
      }
#endif
    }
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(m_mutex);
      stop = 1;
      m_tasks.resize(m_n_threads);
      m_done = m_n_threads;
    }
    m_condition.notify_all();
    for (auto &th : m_threads) th.join();
  }

  ThreadPool& sync() {
    while (m_done > 0)
      std::this_thread::sleep_for(std::chrono::nanoseconds(100));
    m_tasks.clear();
    return *this;
  }

  template <class T>
  ThreadPool& async(std::vector<std::function<T> >& funcs) {
    for (auto& func : funcs) {
      m_tasks.emplace_back([&]() {
        func();
      });
    }
    m_done = funcs.size();
    m_condition.notify_all();
    return *this;
  }
};

}  // namespace tick

#endif  // LIB_INCLUDE_TICK_BASE_PARALLEL_THREAD_POOL_H_
