// License: BSD 3 clause

#include "tick/array/serializer.h"

#define DEBUG_COSTLY_THROW 1
#include <gtest/gtest.h>
#include "tick/base/base.h"

TEST(Serialize, ForceLoadError) {
  // CSR matrix example from https://en.wikipedia.org/wiki/Sparse_matrix
  ArrayFloat data{10, 20, 30, 40, 50, 60, 70, 80};
  Array<INDICE_TYPE> row_indices{0, 2, 4, 7, 8};
  Array<INDICE_TYPE> indices{0, 1, 1, 3, 2, 3, 4, 5};
  SparseArrayFloat2d sparse_array(4, 6, row_indices.data(), indices.data(), data.data());
  bool caught = false;
  try {
    std::string file_name = "test_sparse_array_2d_float.cereal";
    array_to_file(file_name, sparse_array);
    SSparseArrayDouble2dPtr loaded_array = array_from_file<SSparseArrayDouble2d>(file_name);
  } catch (const std::exception &e) {
    caught = true;
  }
  EXPECT_EQ(caught, true);
}

TEST(Serialize, SparseArray2dDouble) {
  // CSR matrix example from https://en.wikipedia.org/wiki/Sparse_matrix
  ArrayDouble data{10, 20, 30, 40, 50, 60, 70, 80};
  Array<INDICE_TYPE> row_indices{0, 2, 4, 7, 8};
  Array<INDICE_TYPE> indices{0, 1, 1, 3, 2, 3, 4, 5};

  SparseArrayDouble2d sparse_array(4, 6, row_indices.data(), indices.data(),
                                   data.data());
  ArrayDouble dot_array{1, 2, 3, 4, 5, 6};

  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(sparse_array, 0)), 50);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(sparse_array, 1)), 220);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(sparse_array, 2)), 740);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(sparse_array, 3)), 480);

  std::string file_name = "test_sparse_array_2d_double.cereal";
  array_to_file(file_name, sparse_array);
  SSparseArrayDouble2dPtr loaded_array =
      array_from_file<SSparseArrayDouble2d>(file_name);

  EXPECT_EQ(loaded_array->n_rows(), 4u);
  EXPECT_EQ(loaded_array->n_cols(), 6u);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(*loaded_array, 0)), 50);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(*loaded_array, 1)), 220);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(*loaded_array, 2)), 740);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(*loaded_array, 3)), 480);

  std::remove(file_name.c_str());
}

TEST(Serialize, SparseArray2dFloat) {
  // CSR matrix example from https://en.wikipedia.org/wiki/Sparse_matrix
  ArrayFloat data{10, 20, 30, 40, 50, 60, 70, 80};
  Array<INDICE_TYPE> row_indices{0, 2, 4, 7, 8};
  Array<INDICE_TYPE> indices{0, 1, 1, 3, 2, 3, 4, 5};

  SparseArrayFloat2d sparse_array(4, 6, row_indices.data(), indices.data(),
                                  data.data());
  ArrayFloat dot_array{1, 2, 3, 4, 5, 6};

  EXPECT_FLOAT_EQ(dot_array.dot(view_row(sparse_array, 0)), 50);
  EXPECT_FLOAT_EQ(dot_array.dot(view_row(sparse_array, 1)), 220);
  EXPECT_FLOAT_EQ(dot_array.dot(view_row(sparse_array, 2)), 740);
  EXPECT_FLOAT_EQ(dot_array.dot(view_row(sparse_array, 3)), 480);

  std::string file_name = "test_sparse_array_2d_float.cereal";
  array_to_file(file_name, sparse_array);
  SSparseArrayFloat2dPtr loaded_array =
      array_from_file<SSparseArrayFloat2d>(file_name);

  EXPECT_EQ(loaded_array->n_rows(), 4u);
  EXPECT_EQ(loaded_array->n_cols(), 6u);
  EXPECT_FLOAT_EQ(dot_array.dot(view_row(*loaded_array, 0)), 50);
  EXPECT_FLOAT_EQ(dot_array.dot(view_row(*loaded_array, 1)), 220);
  EXPECT_FLOAT_EQ(dot_array.dot(view_row(*loaded_array, 2)), 740);
  EXPECT_FLOAT_EQ(dot_array.dot(view_row(*loaded_array, 3)), 480);

  std::remove(file_name.c_str());
}

TEST(Serialize, SparseArray2dAtomicDouble) {
  // CSR matrix example from https://en.wikipedia.org/wiki/Sparse_matrix
  Array<std::atomic<double> > data(8);
  {
    size_t i = 0;
    for (const auto &v : {10, 20, 30, 40, 50, 60, 70, 80})
      data.set_data_index(i++, v);
  }
  Array<INDICE_TYPE> row_indices{0, 2, 4, 7, 8};
  Array<INDICE_TYPE> indices{0, 1, 1, 3, 2, 3, 4, 5};

  SparseArrayAtomicDouble2d sparse_array(4, 6, row_indices.data(),
                                         indices.data(), data.data());
  ArrayDouble dot_array{1, 2, 3, 4, 5, 6};

  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(sparse_array, 0)), 50);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(sparse_array, 1)), 220);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(sparse_array, 2)), 740);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(sparse_array, 3)), 480);

  std::string file_name = "test_sparse_array_2d_double.cereal";
  array_to_file(file_name, sparse_array);
  SSparseArrayDouble2dPtr loaded_array =
      array_from_file<SSparseArrayDouble2d>(file_name);

  EXPECT_EQ(loaded_array->n_rows(), 4u);
  EXPECT_EQ(loaded_array->n_cols(), 6u);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(*loaded_array, 0)), 50);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(*loaded_array, 1)), 220);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(*loaded_array, 2)), 740);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(*loaded_array, 3)), 480);

  std::remove(file_name.c_str());
}

TEST(Serialize, SparseArray2dAtomicFloat) {
  // CSR matrix example from https://en.wikipedia.org/wiki/Sparse_matrix
  Array<std::atomic<float> > data(8);
  {
    size_t i = 0;
    for (const auto &v : {10, 20, 30, 40, 50, 60, 70, 80})
      data.set_data_index(i++, v);
  }
  Array<INDICE_TYPE> row_indices{0, 2, 4, 7, 8};
  Array<INDICE_TYPE> indices{0, 1, 1, 3, 2, 3, 4, 5};
  SparseArrayAtomicFloat2d sparse_array(4, 6, row_indices.data(),
                                        indices.data(), data.data());
  ArrayFloat dot_array{1, 2, 3, 4, 5, 6};

  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(sparse_array, 0)), 50);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(sparse_array, 1)), 220);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(sparse_array, 2)), 740);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(sparse_array, 3)), 480);

  std::string file_name = "test_sparse_array_2d_float.cereal";
  array_to_file(file_name, sparse_array);
  SSparseArrayFloat2dPtr loaded_array =
      array_from_file<SSparseArrayFloat2d>(file_name);

  EXPECT_EQ(loaded_array->n_rows(), 4u);
  EXPECT_EQ(loaded_array->n_cols(), 6u);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(*loaded_array, 0)), 50);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(*loaded_array, 1)), 220);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(*loaded_array, 2)), 740);
  EXPECT_DOUBLE_EQ(dot_array.dot(view_row(*loaded_array, 3)), 480);

  std::remove(file_name.c_str());
}

#ifdef ADD_MAIN
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
