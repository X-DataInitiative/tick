

#ifndef LIB_INCLUDE_TICK_ARRAY_COO_MATRIX_H_
#define LIB_INCLUDE_TICK_ARRAY_COO_MATRIX_H_

template <typename T>
class CooMatrix {
 public:
  CooMatrix() {}

  CooMatrix(const ArrayULong &rows, const ArrayULong &cols, const Array<T> data) {
    this->rows = rows;
    this->cols = cols;
    this->data = data;
    checkCoo();
  }

  explicit CooMatrix(std::shared_ptr<SSparseArray2d<T, RowMajor>> sparse) {
    rows = ArrayULong(sparse->size_sparse());
    nnz = 0;

    std::vector<ulong> nnz_rows;
    for (ulong i = 0; i < sparse->n_rows(); i++) {
      nnz_rows.push_back(sparse->row_indices()[i + 1] - sparse->row_indices()[i]);
    }

    ulong out_i = 0;
    ulong row_i = 0;
    for (ulong nnz_i : nnz_rows) {
      nnz += nnz_i;
      for (ulong i = 0; i < nnz_i; i++) {
        rows[out_i] = row_i;
        out_i++;
        if (out_i > rows.size()) TICK_ERROR("Invalid sparse matrix");
      }
      row_i++;
    }

    auto toArrayULong = [](ArrayUInt &array) {
      ArrayULong out(array.size());
      for (ulong i = 0; i < array.size(); i++) out[i] = (ulong)array[i];
      return out;
    };

    ArrayUInt temp(sparse->size_sparse(), sparse->indices());
    cols = toArrayULong(temp);
    data = Array<T>(sparse->size_sparse(), sparse->data());

    checkCoo();
  }

  void checkCoo() {
    if (rows.size() != cols.size() || cols.size() != data.size() || data.size() != rows.size())
      TICK_ERROR("CooMatrix::checkCoo row, cols, and data size are different");
    // more check?
  }

  void clearZero() {
    checkCoo();

    std::vector<ulong> out_row;
    std::vector<ulong> out_col;
    std::vector<T> out_data;
    for (ulong i = 0; i < rows.size(); i++) {
      if (rows[i] != 0 || cols[i] != 0 || data[i] != (T)0) {
        out_row.push_back(rows[i]);
        out_col.push_back(cols[i]);
        out_data.push_back(data[i]);
      }
    }
    rows = ArrayULong(out_row.size());
    cols = ArrayULong(out_col.size());
    data = Array<T>(out_data.size());

    for (ulong i = 0; i < rows.size(); i++) {
      rows[i] = out_row[i];
      cols[i] = out_col[i];
      data[i] = out_data[i];
    }
  }

  void sortByRow() {
    checkCoo();

    std::vector<std::tuple<ulong, ulong, T>> sort_data;
    for (ulong i = 0; i < rows.size(); i++) sort_data.emplace_back(rows[i], cols[i], data[i]);

    std::sort(sort_data.begin(), sort_data.end());

    for (ulong i = 0; i < rows.size(); i++) {
      rows[i] = std::get<0>(sort_data[i]);
      cols[i] = std::get<1>(sort_data[i]);
      data[i] = std::get<2>(sort_data[i]);
    }

    checkCoo();
  }

  std::shared_ptr<SSparseArray2d<T>> toSparse(ulong n_rows, ulong n_cols) {
    checkCoo();

    clearZero();
    sortByRow();

    std::vector<unsigned int> rows_vec(n_rows + 1);
    std::vector<unsigned int> cols_vec;
    rows_vec[0] = 0;

    std::vector<ulong> nnz_rows;
    for (ulong i = 0; i < n_rows; i++) {
      ulong nnz_this_row = 0;
      for (ulong j = 0; j < rows.size(); j++) {
        if (rows[j] == i) {
          nnz_this_row++;
        }
      }
      nnz_rows.push_back(nnz_this_row);
    }

    if (nnz_rows.size() != n_rows) {
      TICK_ERROR("Unexcepted error nnz_rows.size() != n_rows");
    }

    for (ulong i = 1; i < n_rows + 1; i++) {
      rows_vec[i] = rows_vec[i - 1] + nnz_rows[i - 1];
    }

    ulong maxcol = 0;
    for (ulong i = 0; i < cols.size(); i++) {
      if (cols[i] > maxcol) maxcol = cols[i];
      cols_vec.push_back(cols[i]);
    }

    unsigned int *row_ptr = new unsigned int[rows_vec.size()];
    unsigned int *col_ptr = new unsigned int[cols_vec.size()];
    T *data_ptr = new T[data.size()];

    memcpy(row_ptr, rows_vec.data(), rows_vec.size() * sizeof(unsigned int));
    memcpy(col_ptr, cols_vec.data(), cols_vec.size() * sizeof(unsigned int));
    memcpy(data_ptr, data.data(), data.size() * sizeof(T));

    std::shared_ptr<SSparseArray2d<T>> arrayptr = SSparseArray2d<T>::new_ptr(0, 0, 0);

    arrayptr->set_data_indices_rowindices(data_ptr, col_ptr, row_ptr, n_rows, n_cols);
    return arrayptr;
  }

  ArrayULong rows;
  ArrayULong cols;
  Array<T> data;
  ulong nnz;
};

#endif  // LIB_INCLUDE_TICK_ARRAY_COO_MATRIX_H_
