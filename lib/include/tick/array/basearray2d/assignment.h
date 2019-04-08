
#ifndef LIB_INCLUDE_TICK_ARRAY_BASEARRAY2D_ASSIGNMENT_H_
#define LIB_INCLUDE_TICK_ARRAY_BASEARRAY2D_ASSIGNMENT_H_


// Assignement operator.
// It copies the data creating new allocation owned by the new array
template <typename T, typename MAJ>
BaseArray2d<T, MAJ>& BaseArray2d<T, MAJ>::operator=(const BaseArray2d<T, MAJ> &that) {
  if (this != &that) {
    AbstractArray1d2d<T, MAJ>::operator=(that);
    if (is_row_indices_allocation_owned && _row_indices != nullptr)
      TICK_PYTHON_FREE(_row_indices);
    _row_indices = nullptr;
    is_row_indices_allocation_owned = true;
    _n_cols = that._n_cols;
    _n_rows = that._n_rows;
    _size = _n_cols * _n_rows;
    if (that.is_sparse()) {
      TICK_PYTHON_MALLOC(_row_indices, INDICE_TYPE, _n_rows + 1);
      memcpy(_row_indices, that._row_indices,
             sizeof(INDICE_TYPE) * (_n_rows + 1));
    }
  }
  return (*this);
}

// greatfully stolen from scipy sparsetools
// https://github.com/scipy/scipy/blob/master/scipy/sparse/sparsetools/csr.h
namespace tick {
template <class I, class T>
void csr_tocsc(const size_t n_row,
               const size_t n_col,
               const I Ap[],
               const I Aj[],
               const T Ax[],
                     I Bp[],
                     I Bi[],
                     T Bx[]) {
    const I nnz = Ap[n_row];

    std::fill(Bp, Bp + n_col, 0);

    for (I n = 0; n < nnz; n++) {
        Bp[Aj[n]]++;
    }

    for (I col = 0, cumsum = 0; col < n_col; col++) {
        I temp  = Bp[col];
        Bp[col] = cumsum;
        cumsum += temp;
    }
    Bp[n_col] = nnz;

    for (I row = 0; row < n_row; row++) {
        for (I jj = Ap[row]; jj < Ap[row+1]; jj++) {
            I col  = Aj[jj];
            I dest = Bp[col];

            Bi[dest] = row;
            Bx[dest] = Ax[jj];

            Bp[col]++;
        }
    }

    for (I col = 0, last = 0; col <= n_col; col++) {
        I temp  = Bp[col];
        Bp[col] = last;
        last    = temp;
    }
}

template <class I, class T>
void csc_tocsr(const size_t n_row,
               const size_t n_col,
               const I Ap[],
               const I Ai[],
               const T Ax[],
                     I Bp[],
                     I Bj[],
                     T Bx[]) {
  csr_tocsc<I, T>(n_col, n_row, Ap, Ai, Ax, Bp, Bj, Bx);
}
}  // namespace tick

// Assignement operator from row major to col major.
template <typename T, typename MAJ>
template <typename RIGHT_MAJ>
typename std::enable_if<std::is_same<MAJ, ColMajor>::value && std::is_same<RIGHT_MAJ, RowMajor>::value, BaseArray2d<T, MAJ>>::type&
BaseArray2d<T, MAJ>::operator=(const BaseArray2d<T, RIGHT_MAJ> &that) {
  AbstractArray1d2d<T, MAJ>::operator=(that);
  if (is_row_indices_allocation_owned && _row_indices != nullptr)
    TICK_PYTHON_FREE(_row_indices);
  _row_indices = nullptr;
  is_row_indices_allocation_owned = true;
  _n_rows = that.n_rows();
  _n_cols = that.n_cols();
  _size = _n_cols * _n_rows;
  if (that.is_sparse()) {
    TICK_PYTHON_MALLOC(_row_indices, INDICE_TYPE, _n_cols + 1);
    _row_indices[_n_cols] = _size_sparse;
    tick::csr_tocsc(
      _n_rows, _n_cols,
      that.row_indices(), that.indices(), that.data(),
      _row_indices, _indices, _data);
  } else {
    for (size_t i = 0; i < _n_cols; i++)
      for (size_t j = 0; j < _n_rows; j++)
        _data[ i * _n_rows + j ] = that.data()[ j * _n_cols + i ];
  }

  return (*this);
}

// Assignement operator from col major to row major.
template <typename T, typename MAJ>
template <typename RIGHT_MAJ>
typename std::enable_if<std::is_same<MAJ, RowMajor>::value && std::is_same<RIGHT_MAJ, ColMajor>::value, BaseArray2d<T, MAJ>>::type&
BaseArray2d<T, MAJ>::operator=(const BaseArray2d<T, RIGHT_MAJ> &that) {
  AbstractArray1d2d<T, MAJ>::operator=(that);
  if (is_row_indices_allocation_owned && _row_indices != nullptr)
    TICK_PYTHON_FREE(_row_indices);
  _row_indices = nullptr;
  is_row_indices_allocation_owned = true;
  _n_cols = that.n_cols();
  _n_rows = that.n_rows();
  _size = _n_cols * _n_rows;
  if (that.is_sparse()) {
    TICK_PYTHON_MALLOC(_row_indices, INDICE_TYPE, _n_rows + 1);
    tick::csc_tocsr(
      _n_rows, _n_cols,
      that.row_indices(), that.indices(), that.data(),
      _row_indices, _indices, _data);
  } else {
    for (size_t i = 0; i < _n_cols; i++)
      for (size_t j = 0; j < _n_rows; j++)
        _data[ i * _n_cols + j ] = that.data()[ j * _n_rows + i ];
  }

  return (*this);
}



#endif  // LIB_INCLUDE_TICK_ARRAY_BASEARRAY2D_ASSIGNMENT_H_
