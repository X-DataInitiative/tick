// License: BSD 3 clause

#include <csignal>

#include <numeric>
#include <algorithm>
#include <complex>
#include <fstream>

#define DEBUG_COSTLY_THROW 1
#define XDATA_TEST_DATA_SIZE (1000)

#include "tick/base/parallel/parallel.h"
#include "tick/base/time_func.h"
#include "tick/array/array2d.h"

#include <gtest/gtest.h>
#include <cereal/archives/json.hpp>


struct MapFunctorsUnary {
  MapFunctorsUnary(std::size_t n)
      : data(n, 0) {}

  unsigned long Set(unsigned long i) {
    data[i] = i;

    return i;
  }

  void Double(unsigned long i) {
    data[i] *= 2;
  }

  void Scale(unsigned long i, long alpha) {
    data[i] *= alpha;
  }

  std::vector<long> data;
};

class ParallelTest : public ::testing::TestWithParam<unsigned> {};

TEST_P(ParallelTest, Set) {
  const std::size_t n{XDATA_TEST_DATA_SIZE};

  MapFunctorsUnary m{n};

  std::vector<long> expected(n);

  ASSERT_EQ(expected, m.data);

  std::iota(std::begin(expected), std::end(expected), 0);
  parallel_run(GetParam(), n, &MapFunctorsUnary::Set, &m);

  EXPECT_EQ(expected, m.data);
}

TEST_P(ParallelTest, Double) {
  const std::size_t n{XDATA_TEST_DATA_SIZE};

  MapFunctorsUnary m{n};
  std::iota(std::begin(m.data), std::end(m.data), 0);

  std::vector<long> expected = m.data;

  ASSERT_EQ(expected, m.data);

  for (auto &x : expected) x *= 2;

  parallel_run(GetParam(), n, &MapFunctorsUnary::Double, &m);

  EXPECT_EQ(expected, m.data);
}

TEST_P(ParallelTest, Scale) {
  const std::size_t n{XDATA_TEST_DATA_SIZE};

  MapFunctorsUnary m{n};

  std::vector<long> expected{};
  for (auto a : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
    std::iota(std::begin(m.data), std::end(m.data), 0);

    expected = m.data;

    ASSERT_EQ(expected, m.data);

    for (auto &x : expected) x *= a;

    parallel_run(GetParam(), n, &MapFunctorsUnary::Scale, &m, a);

    EXPECT_EQ(expected, m.data);
  }
}

long plus_f(const long a, const long b) { return a + b; }

TEST_P(ParallelTest, ReduceSum) {
  const std::size_t n{XDATA_TEST_DATA_SIZE};

  MapFunctorsUnary m{n};

  const long result = parallel_map_reduce(GetParam(), m.data.size(), plus_f, &MapFunctorsUnary::Set, &m);

  const auto na = n - 1;
  EXPECT_EQ((na * (na + 1)) / 2, result);
}

TEST_P(ParallelTest, ReduceSumAdditive) {
  const std::size_t n{XDATA_TEST_DATA_SIZE};

  MapFunctorsUnary m{n};

  const long result = parallel_map_additive_reduce(GetParam(), m.data.size(), &MapFunctorsUnary::Set, &m);

  const auto na = n - 1;
  EXPECT_EQ((na * (na + 1)) / 2, result);
}

struct CalcFibo {

  unsigned long Fibo(unsigned long n, unsigned long u0, unsigned long u1) {
    unsigned long a, b;

    a = u0;
    b = u1;

    while (n > 0) {
      unsigned long temp = a + b;

      a = b;
      b = temp;

      --n;
    }

    return a;
  }

  unsigned long DoIt(unsigned long i) {
    return Fibo(i, 0, 1);
  }

};

TEST_P(ParallelTest, MapFibo) {
  CalcFibo c{};
  auto result = parallel_map(GetParam(), 100, &CalcFibo::DoIt, &c);

  std::vector<unsigned long> resultAsVector{result->data(), result->data() + result->size()};
  std::vector<unsigned long> expected(resultAsVector.size(), 0);

  {
    std::size_t i = 0;
    std::generate(std::begin(expected), std::end(expected), [&i] { return CalcFibo{}.Fibo(i++, 0, 1); });
  }

  EXPECT_EQ(expected, resultAsVector);
}

struct ComplexFunctors {
  std::complex<double> DoIt(unsigned long i) {
    return std::proj(std::sqrt(std::cos(std::sin(std::complex<double>(0, 1)))))
        * std::exp(std::complex<double>(0.0, 1.0) * std::acos(-1));
  }
};

TEST_P(ParallelTest, MapComplex) {
  ComplexFunctors c{};

  std::vector<std::complex<double>> result = parallel_map(GetParam(), 100000, &ComplexFunctors::DoIt, &c);
  std::vector<std::complex<double>> expected(result.size(), 0);

  {
    std::size_t i = 0;
    std::generate(std::begin(expected), std::end(expected), [&] { return c.DoIt(i++); });
  }

  EXPECT_EQ(expected, result);
}

struct ExceptionThrower {
  void DoIt(unsigned long i) {
    throw std::runtime_error("Example");
  }
};

struct ExceptionThrowerBadIndex {
  void DoIt(unsigned long i, ArrayDouble &arrayDouble) {
    arrayDouble[i] += 1;
  }
};

TEST_P(ParallelTest, ExceptionThrow) {
  ExceptionThrower e{};

  EXPECT_THROW(parallel_run(GetParam(), 1000, &ExceptionThrower::DoIt, &e), std::runtime_error);
}

TEST_P(ParallelTest, ExceptionThrowBadIndex) {
  ExceptionThrowerBadIndex e{};

  ArrayDouble arrayDouble(1);
  arrayDouble.fill(1.0);

  EXPECT_THROW(parallel_run(GetParam(), 1000, &ExceptionThrowerBadIndex::DoIt, &e, arrayDouble), std::exception);
}

struct SignalRaiser {
  void DoIt(unsigned long i) {
    if (!Interruption::is_raised())
      std::raise(SIGINT);
  }
};

TEST_P(ParallelTest, SignalInterrupt) {
  SignalRaiser s{};

  EXPECT_THROW(parallel_run(GetParam(), 1000, &SignalRaiser::DoIt, &s), Interruption);

  Interruption::reset();
}

TEST_P(ParallelTest, MapArray) {
  const std::size_t N{1000};

  ArrayDouble data(N);
  data.fill(0.0);

  auto f = [](ulong i, ArrayDouble &s) { s[i] = i; };
  auto redux = [](ArrayDouble &r, ArrayDouble &s) { r.mult_incr(s, 1.0); };

  EXPECT_NO_THROW(parallel_map_array<ArrayDouble>(GetParam(), N, redux, f, data));

  {
    std::vector<double> expected(1000);

    std::size_t i = 0;
    std::generate(std::begin(expected), std::end(expected), [&i]() { return i++; });

    EXPECT_TRUE(std::equal(std::begin(expected), std::end(expected), data.data()));
  }
}

INSTANTIATE_TEST_CASE_P(AllParallelTests,
                        ParallelTest,
                        ::testing::Values(1, 2, 4, 8, 16));

TEST(ParallelTest, CPUCount) {
  ASSERT_GE(std::thread::hardware_concurrency(), 2);
}

TEST(ParallelTest, TooManyThreads) {
  CalcFibo c;

  parallel_run(8, 4, &CalcFibo::DoIt, &c);
}

TEST(DebugTest, WarningDebug) {
  testing::internal::CaptureStdout();

  TICK_DEBUG() << "Sample debug message";

  const std::string out = testing::internal::GetCapturedStdout();
  EXPECT_GE(out.length(), 0);
  EXPECT_PRED_FORMAT2(testing::IsSubstring, "Sample debug message", out);

  testing::internal::CaptureStderr();

  TICK_WARNING() << "Sample warning";

  const std::string err = testing::internal::GetCapturedStderr();
  EXPECT_GE(err.length(), 0);
  EXPECT_PRED_FORMAT2(testing::IsSubstring, "Sample warning", err);

  EXPECT_THROW(TICK_ERROR("Sample"), std::runtime_error);
  EXPECT_THROW(TICK_BAD_INDEX(0, 10, 100), std::out_of_range);
}

TEST(DebugTest, PrintArray) {
  testing::internal::CaptureStdout();

  ArrayDouble emptyArray(0);
  TICK_DEBUG() << "emptyArray: " << emptyArray;

  ArrayDouble d(10);
  for (ulong i = 0; i < d.size(); ++i) d[i] = i;

  TICK_DEBUG() << "d: " << d;

  ArrayDouble dBig(100);
  for (ulong i = 0; i < dBig.size(); ++i) dBig[i] = i;

  TICK_DEBUG() << "dBig: " << dBig;

  testing::internal::GetCapturedStdout();
}

TEST(TimeFuncTest, Serialization) {
  TimeFunction tf;
  ArrayDouble T({0.0, 1.0, 2.0});
  ArrayDouble Y({1.0, 0.0, -1.0});

  tf = TimeFunction(T, Y, 0.2);

  std::stringstream ss;
  {
    cereal::JSONOutputArchive outputArchive(ss);

    outputArchive(tf);
  }

  {
    cereal::JSONInputArchive inputArchive(ss);

    TimeFunction tf_restored(0.0);
    inputArchive(tf_restored);

    ASSERT_DOUBLE_EQ(tf.value(0.0), tf_restored.value(0.0));
    ASSERT_DOUBLE_EQ(tf.value(1.0), tf_restored.value(1.0));
    ASSERT_DOUBLE_EQ(tf.value(0.5), tf_restored.value(0.5));
    ASSERT_DOUBLE_EQ(tf.value(1.5), tf_restored.value(1.5));
  }
}

TEST(DebugTest, PrintArray2D) {
  testing::internal::CaptureStdout();

  ArrayDouble2d arr(10, 10);
  TICK_DEBUG() << "arr: " << arr;

  const std::string msg = testing::internal::GetCapturedStdout();
  EXPECT_PRED_FORMAT2(testing::IsSubstring, "Array2D", msg);
}

TEST(DebugTest, PrintSparseArray) {
  testing::internal::CaptureStdout();

  INDICE_TYPE indices[] = {1, 4, 5, 7, 8, 16};
  double data[] = {0, 1, 2, 3, 4, 5};
  SparseArrayDouble sparseArrayDouble(20, 6, indices, data);

  TICK_DEBUG() << sparseArrayDouble;

  const std::string msg = testing::internal::GetCapturedStdout();
  EXPECT_PRED_FORMAT2(testing::IsSubstring, "SparseArray", msg);
}

#ifdef ADD_MAIN
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
