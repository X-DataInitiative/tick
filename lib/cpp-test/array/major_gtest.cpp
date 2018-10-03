// License: BSD 3 clause

#include <algorithm>

#define DEBUG_COSTLY_THROW 1

#include <gtest/gtest.h>
#include "tick/array/sparsearray2d.h"
#include "tick/base/base.h"

TEST(SparseArray2d, MAJOR) {
  {
    Array<double> data{10, 20, 30, 40, 50, 60, 70, 80};
    Array<INDICE_TYPE> row_indices{0, 2, 4, 7, 8};
    Array<INDICE_TYPE> indices{0, 1, 1, 3, 2, 3, 4, 5};
    SparseArray2d<double> sparse_array(4, 6, row_indices.data(), indices.data(),
                                     data.data());
    auto col_sparse = SparseArray2d<double, ColMajor>::CREATE_FROM(sparse_array);
    auto row_sparse = SparseArray2d<double>::CREATE_FROM(col_sparse);
    ASSERT_TRUE(sparse_array == row_sparse);
  }
  {
    Array<float> data{10, 20, 30, 40, 50, 60, 70, 80};
    Array<INDICE_TYPE> row_indices{0, 2, 4, 7, 8};
    Array<INDICE_TYPE> indices{0, 1, 1, 3, 2, 3, 4, 5};
    SparseArray2d<float> sparse_array(4, 6, row_indices.data(), indices.data(),
                                     data.data());
    auto col_sparse = SparseArray2d<float, ColMajor>::CREATE_FROM(sparse_array);
    auto row_sparse = SparseArray2d<float>::CREATE_FROM(col_sparse);
    ASSERT_TRUE(sparse_array == row_sparse);
  }
}

TEST(SparseArray2d, MAJOR_view_row) {
  {
    // Array representation
    // 10 20  0  0  0  0
    //  0 30  0 40  0  0
    //  0  0 50 60 70  0
    //  0  0  0  0  0 80
    Array<double> data{10, 20, 30, 40, 50, 60, 70, 80};
    Array<INDICE_TYPE> row_indices{0, 2, 4, 7, 8};
    Array<INDICE_TYPE> indices{0, 1, 1, 3, 2, 3, 4, 5};
    SparseArray2d<double> sparse_array(4, 6, row_indices.data(), indices.data(),
                                       data.data());
    Array<double> row_1_data{30, 40};
    Array<INDICE_TYPE> row_1_indices{1, 3};
    SparseArray<double> row_1(6, 2, row_1_indices.data(), row_1_data.data());

    ASSERT_TRUE(view_row(sparse_array, 1) == row_1);

    auto col_sparse = SparseArray2d<double, ColMajor>::CREATE_FROM(sparse_array);

    Array<double> col_3_data{40, 60};
    Array<INDICE_TYPE> col_3_indices{1, 2};
    SparseArray<double> col_3(4, 2, col_3_indices.data(), col_3_data.data());

    ASSERT_TRUE(view_col(col_sparse, 3) == col_3);
  }
  {
    Array<float> data{10, 20, 30, 40, 50, 60, 70, 80};
    Array<INDICE_TYPE> row_indices{0, 2, 4, 7, 8};
    Array<INDICE_TYPE> indices{0, 1, 1, 3, 2, 3, 4, 5};
    SparseArray2d<float> sparse_array(4, 6, row_indices.data(), indices.data(),
                                       data.data());
    Array<float> row_1_data{30, 40};
    Array<INDICE_TYPE> row_1_indices{1, 3};
    SparseArray<float> row_1(6, 2, row_1_indices.data(), row_1_data.data());

    ASSERT_TRUE(view_row(sparse_array, 1) == row_1);

    auto col_sparse = SparseArray2d<float, ColMajor>::CREATE_FROM(sparse_array);

    Array<float> col_3_data{40, 60};
    Array<INDICE_TYPE> col_3_indices{1, 2};
    SparseArray<float> col_3(4, 2, col_3_indices.data(), col_3_data.data());

    ASSERT_TRUE(view_col(col_sparse, 3) == col_3);
  }
}

TEST(Array2d, MAJOR_view_row) {
  const ulong n_rows = 4;
  const ulong n_cols = 3;
  {
    Array<double> row_data{11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43};
    Array2d<double> row_major_array(n_rows, n_cols, row_data.data());
    Array<double> row_1 {21, 22, 23};
    ASSERT_TRUE(view_row(row_major_array, 1) == row_1);

    Array<double> col_data{11, 21, 31, 41, 12, 22, 32, 42, 13, 23, 33, 43};
    Array2d<double, ColMajor> col_major_array(n_rows, n_cols, col_data.data());
    Array<double> col_1 {12, 22, 32, 42};
    ASSERT_TRUE(view_col(col_major_array, 1) == col_1);
  }
  {
    Array<float> row_data{11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43};
    Array2d<float> row_major_array(n_rows, n_cols, row_data.data());
    Array<float> row_1 {21, 22, 23};
    ASSERT_TRUE(view_row(row_major_array, 1) == row_1);

    Array<float> col_data{11, 21, 31, 41, 12, 22, 32, 42, 13, 23, 33, 43};
    Array2d<float, ColMajor> col_major_array(n_rows, n_cols, col_data.data());
    Array<float> col_1 {12, 22, 32, 42};
    ASSERT_TRUE(view_col(col_major_array, 1) == col_1);
  }
}

#ifdef ADD_MAIN
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
