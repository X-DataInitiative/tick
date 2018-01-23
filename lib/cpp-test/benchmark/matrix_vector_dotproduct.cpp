
#include "tick/base/base.h"
#include "tick/random/test_rand.h"

int main(int nargs, char **args) {
  ulong n_rows = 20000;
  ulong n_cols = 10000;

  if (nargs > 1)
    n_rows = std::stoul(args[1]);

  if (nargs > 2)
    n_cols = std::stoul(args[2]);

  const ulong num_runs = 5;
  const ulong num_iterations = 10;

  const auto sample = test_uniform(n_rows * n_cols);
  ArrayDouble2d matrix(n_rows, n_cols, sample->data());
  const ArrayDouble vector = *test_uniform(n_cols);

  ArrayDouble out(n_rows);

  for (ulong run_i = 0; run_i < num_runs; ++run_i) {

    const auto start = std::chrono::system_clock::now();
    for (ulong iter = 0; iter < num_iterations; ++iter) {

      for (ulong i = 0; i < n_rows; ++i) {
        out[i] = view_row(matrix, i).dot(vector);
      }
    }

    const auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << elapsed_seconds.count() << '\t'
              << num_iterations << '\t'
              << n_rows << '\t'
              << n_cols << '\t'
              << args[0] << '\t'
              << std::endl;
  }
}