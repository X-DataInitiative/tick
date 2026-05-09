
#ifndef LIB_INCLUDE_TICK_ARRAY_SPARSE2D_RANDOM2D_H_
#define LIB_INCLUDE_TICK_ARRAY_SPARSE2D_RANDOM2D_H_

#include <random>

template <typename T, typename MAJ>
std::shared_ptr<SparseArray2d<T, MAJ>> SparseArray2d<T, MAJ>::RANDOM(size_t rows, size_t cols, T density, T seed) {
    if (density < 0 || density > 1)
      throw std::runtime_error("Invalid sparse density, must be between 0 and 1");

    size_t size = std::floor(rows * cols * density);
    auto arr = SSparseArray2d<T, MAJ>::new_ptr(rows, cols, size);

    std::mt19937_64 generator;
    if (seed > 0) {
      generator = std::mt19937_64(seed);
    } else {
      std::random_device r;
      std::seed_seq seed_seq{r(), r(), r(), r(), r(), r(), r(), r()};
      generator = std::mt19937_64(seed_seq);
    }
    std::uniform_real_distribution<T> dist;
    auto data =  arr->data();
    for (size_t i = 0; i < size; i++) data[i] = dist(generator);

    size_t nnz = size;
    std::vector<size_t> nnz_row(rows, 0);

    size_t index = 0;
    while (nnz > 0) {
      std::uniform_int_distribution<size_t> dist_int(1, 100);  // to do 50 50
      if (dist_int(generator) > 50) {
        nnz_row[index]++;
        nnz--;
      }
      index++;
      if (index >= rows) index = 0;
    }

    index = 0;
    auto indices = arr->indices();
    for (size_t i : nnz_row) {
      std::vector<size_t> indice_comb;
      for (size_t j = 0; j < cols; j++) indice_comb.emplace_back(j);
      std::shuffle(indice_comb.begin(), indice_comb.end(), generator);
      for (size_t j = 0; j < i; j++) {
        indices[index++] = indice_comb[j];
      }
    }

    // if (index != arr->indices().size() - 1)
    //   std::runtime_error("Uh something is wrong");

    auto row_indices = arr->row_indices();
    row_indices[0] = 0;
    for (size_t i = 1; i < rows + 1; i++) row_indices[i] = row_indices[i - 1] + nnz_row[i - 1];

    return arr;
  }

#endif  // LIB_INCLUDE_TICK_ARRAY_SPARSE2D_RANDOM2D_H_
