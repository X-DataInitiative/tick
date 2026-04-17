#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <numpy/arrayobject.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tick/array/array.h"
#include "tick/array/array2d.h"
#include "tick/array/basearray.h"
#include "tick/array/basearray2d.h"
#include "tick/array/sarray.h"
#include "tick/array/sarray2d.h"
#include "tick/array/sbasearray.h"
#include "tick/array/sbasearray2d.h"
#include "tick/array/sparsearray.h"
#include "tick/array/sparsearray2d.h"
#include "tick/array/ssparsearray.h"
#include "tick/array/ssparsearray2d.h"
#include "tick/array/varray.h"
#include "tick/base/serialization.h"

namespace py = pybind11;

namespace tick::pybind {

inline bool ensure_numpy_imported() {
  static bool imported = false;
  if (!imported) {
    if (_import_array() < 0) {
      PyErr_Clear();
      return false;
    }
    imported = true;
  }
  return true;
}

template <typename Value, typename Class>
void enable_cereal_pickle(Class &cls) {
  cls.def(py::pickle(
      [](const Value &value) {
        return py::make_tuple(
            tick::object_to_string(const_cast<Value *>(&value)));
      },
      [](py::tuple state) {
        if (state.size() != 1) {
          throw std::runtime_error("Invalid pickle state");
        }
        auto value = std::make_shared<Value>();
        tick::object_from_string(value.get(), state[0].cast<std::string>());
        return value;
      }));
}

template <typename T>
struct numpy_type;

template <>
struct numpy_type<double> {
  static constexpr int value = NPY_DOUBLE;
  static constexpr const char *name = "double";
};

template <>
struct numpy_type<float> {
  static constexpr int value = NPY_FLOAT;
  static constexpr const char *name = "float";
};

template <>
struct numpy_type<std::int16_t> {
  static constexpr int value = NPY_INT16;
  static constexpr const char *name = "short";
};

template <>
struct numpy_type<std::uint16_t> {
  static constexpr int value = NPY_UINT16;
  static constexpr const char *name = "ushort";
};

template <>
struct numpy_type<std::int32_t> {
  static constexpr int value = NPY_INT32;
  static constexpr const char *name = "int";
};

template <>
struct numpy_type<std::uint32_t> {
  static constexpr int value = NPY_UINT32;
  static constexpr const char *name = "uint";
};

template <>
struct numpy_type<std::int64_t> {
  static constexpr int value = NPY_INT64;
  static constexpr const char *name = "long";
};

template <>
struct numpy_type<std::uint64_t> {
  static constexpr int value = NPY_UINT64;
  static constexpr const char *name = "ulong";
};

template <typename T>
inline bool dtype_matches(const py::array &array) {
  return PyArray_TYPE(reinterpret_cast<PyArrayObject *>(array.ptr())) ==
         numpy_type<T>::value;
}

inline bool index_dtype_matches(const py::array &array) {
  auto *array_obj = reinterpret_cast<PyArrayObject *>(array.ptr());
  const int numpy_type_num = PyArray_TYPE(array_obj);
  return PyTypeNum_ISINTEGER(numpy_type_num) &&
         PyArray_ITEMSIZE(array_obj) == sizeof(INDICE_TYPE);
}

template <typename T>
inline bool index_values_fit(const py::array &array) {
  auto *array_obj = reinterpret_cast<PyArrayObject *>(array.ptr());
  const auto *data = static_cast<const T *>(PyArray_DATA(array_obj));
  const auto size = PyArray_SIZE(array_obj);
  constexpr unsigned long long max_index =
      static_cast<unsigned long long>(std::numeric_limits<INDICE_TYPE>::max());

  for (npy_intp i = 0; i < size; ++i) {
    if constexpr (std::is_signed<T>::value) {
      if (data[i] < 0) return false;
    }

    using unsigned_t = typename std::make_unsigned<T>::type;
    if (static_cast<unsigned long long>(static_cast<unsigned_t>(data[i])) >
        max_index) {
      return false;
    }
  }

  return true;
}

inline bool sparse_index_values_fit(const py::array &array) {
  switch (PyArray_TYPE(reinterpret_cast<PyArrayObject *>(array.ptr()))) {
    case NPY_INT8:
      return index_values_fit<std::int8_t>(array);
    case NPY_INT16:
      return index_values_fit<std::int16_t>(array);
    case NPY_INT32:
      return index_values_fit<std::int32_t>(array);
    case NPY_INT64:
      return index_values_fit<std::int64_t>(array);
    case NPY_UINT8:
      return index_values_fit<std::uint8_t>(array);
    case NPY_UINT16:
      return index_values_fit<std::uint16_t>(array);
    case NPY_UINT32:
      return index_values_fit<std::uint32_t>(array);
    case NPY_UINT64:
      return index_values_fit<std::uint64_t>(array);
    default:
      return false;
  }
}

inline bool load_sparse_index_array(const py::handle &src, py::array &out,
                                    py::object &owner) {
  if (!py::isinstance<py::array>(src)) return false;

  py::array array = py::reinterpret_borrow<py::array>(src);
  auto *array_obj = reinterpret_cast<PyArrayObject *>(array.ptr());
  if (!PyTypeNum_ISINTEGER(PyArray_TYPE(array_obj))) return false;

  auto *prepared_obj = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(src.ptr(), PyArray_TYPE(array_obj), NPY_ARRAY_CARRAY_RO));
  if (prepared_obj == nullptr) return false;

  py::object prepared_owner =
      py::reinterpret_steal<py::object>(reinterpret_cast<PyObject *>(prepared_obj));
  py::array prepared = py::reinterpret_borrow<py::array>(prepared_owner);
  if (!sparse_index_values_fit(prepared)) {
    PyErr_SetString(PyExc_ValueError,
                    "Sparse matrix indices exceed the supported index range");
    return false;
  }

  if (index_dtype_matches(prepared)) {
    owner = std::move(prepared_owner);
    out = py::reinterpret_borrow<py::array>(owner);
    return true;
  }

  auto *converted_obj = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
      prepared.ptr(), numpy_type<INDICE_TYPE>::value,
      NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST));
  if (converted_obj == nullptr) return false;

  owner = py::reinterpret_steal<py::object>(reinterpret_cast<PyObject *>(converted_obj));
  out = py::reinterpret_borrow<py::array>(owner);
  return true;
}

template <typename MAJ>
struct major_traits;

template <>
struct major_traits<RowMajor> {
  static constexpr bool fortran = false;
};

template <>
struct major_traits<ColMajor> {
  static constexpr bool fortran = true;
};

inline bool is_scipy_sparse(const py::handle &obj) {
  return py::hasattr(obj, "data") && py::hasattr(obj, "indices") &&
         py::hasattr(obj, "indptr") && py::hasattr(obj, "shape");
}

template <typename T>
inline bool load_dense_array_1d(const py::handle &src, Array<T> &out,
                                py::object &owner) {
  if (!py::isinstance<py::array>(src)) return false;

  py::array array = py::reinterpret_borrow<py::array>(src);
  if (array.ndim() != 1) {
    PyErr_SetString(PyExc_ValueError, "Expecting a 1 dimensional contiguous numpy array");
    return false;
  }
  auto *array_obj = reinterpret_cast<PyArrayObject *>(array.ptr());
  if (!PyArray_ISCARRAY(array_obj) || !PyArray_ISWRITEABLE(array_obj)) {
    PyErr_SetString(PyExc_ValueError, "Expecting a contiguous writable dense numpy array");
    return false;
  }
  if (!dtype_matches<T>(array)) {
    PyErr_SetString(PyExc_ValueError, ("Expecting a " + std::string(numpy_type<T>::name) +
                                       " numpy array").c_str());
    return false;
  }

  owner = py::reinterpret_borrow<py::object>(src);
  out = Array<T>(static_cast<ulong>(array.shape(0)),
                 static_cast<T *>(PyArray_DATA(array_obj)));
  return true;
}

template <typename T, typename MAJ = RowMajor>
inline bool load_dense_array_2d(const py::handle &src, Array2d<T, MAJ> &out,
                                py::object &owner) {
  if (!py::isinstance<py::array>(src)) return false;

  py::array array = py::reinterpret_borrow<py::array>(src);
  if (array.ndim() != 2) {
    PyErr_SetString(PyExc_ValueError, "Expecting a 2 dimensional contiguous numpy array");
    return false;
  }
  auto *array_obj = reinterpret_cast<PyArrayObject *>(array.ptr());
  const bool contiguous = major_traits<MAJ>::fortran
                              ? (PyArray_IS_F_CONTIGUOUS(array_obj) &&
                                 PyArray_ISWRITEABLE(array_obj))
                              : (PyArray_IS_C_CONTIGUOUS(array_obj) &&
                                 PyArray_ISWRITEABLE(array_obj));
  if (!contiguous) {
    PyErr_SetString(PyExc_ValueError, "Expecting a contiguous writable 2 dimensional numpy array");
    return false;
  }
  if (!dtype_matches<T>(array)) {
    PyErr_SetString(PyExc_ValueError, ("Expecting a " + std::string(numpy_type<T>::name) +
                                       " numpy array").c_str());
    return false;
  }

  owner = py::reinterpret_borrow<py::object>(src);
  out = Array2d<T, MAJ>(static_cast<ulong>(array.shape(0)),
                        static_cast<ulong>(array.shape(1)),
                        static_cast<T *>(PyArray_DATA(array_obj)));
  return true;
}

template <typename T>
inline bool load_sparse_array_1d(const py::handle &src, SparseArray<T> &out,
                                 py::object &data_owner,
                                 py::object &indices_owner) {
  if (!is_scipy_sparse(src)) return false;

  py::tuple shape = py::reinterpret_borrow<py::tuple>(src.attr("shape"));
  if (shape.size() != 2 || py::cast<py::ssize_t>(shape[0]) != 1) {
    PyErr_SetString(PyExc_ValueError, "Expecting a 1d sparse array");
    return false;
  }

  py::array data = py::reinterpret_borrow<py::array>(src.attr("data"));
  py::array indices;
  if (!dtype_matches<T>(data) ||
      !load_sparse_index_array(src.attr("indices"), indices, indices_owner)) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_ValueError, ("Expecting a sparse " +
                                         std::string(numpy_type<T>::name) +
                                         " array").c_str());
    }
    return false;
  }
  data_owner = py::reinterpret_borrow<py::object>(src.attr("data"));
  out = SparseArray<T>(static_cast<ulong>(py::cast<py::ssize_t>(shape[1])),
                       static_cast<ulong>(data.shape(0)),
                       static_cast<INDICE_TYPE *>(indices.mutable_data()),
                       static_cast<T *>(data.mutable_data()));
  return true;
}

template <typename T, typename MAJ = RowMajor>
inline bool load_sparse_array_2d(const py::handle &src, SparseArray2d<T, MAJ> &out,
                                 py::object &data_owner,
                                 py::object &indices_owner,
                                 py::object &indptr_owner) {
  if (!is_scipy_sparse(src)) return false;

  py::tuple shape = py::reinterpret_borrow<py::tuple>(src.attr("shape"));
  if (shape.size() != 2) {
    PyErr_SetString(PyExc_ValueError, "Expecting a 2d sparse array");
    return false;
  }

  py::array data = py::reinterpret_borrow<py::array>(src.attr("data"));
  py::array indices;
  py::array indptr;
  if (!dtype_matches<T>(data) ||
      !load_sparse_index_array(src.attr("indices"), indices, indices_owner) ||
      !load_sparse_index_array(src.attr("indptr"), indptr, indptr_owner)) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_ValueError, ("Expecting a sparse " +
                                         std::string(numpy_type<T>::name) +
                                         " array").c_str());
    }
    return false;
  }
  data_owner = py::reinterpret_borrow<py::object>(src.attr("data"));
  out = SparseArray2d<T, MAJ>(
      static_cast<ulong>(py::cast<py::ssize_t>(shape[0])),
      static_cast<ulong>(py::cast<py::ssize_t>(shape[1])),
      static_cast<INDICE_TYPE *>(indptr.mutable_data()),
      static_cast<INDICE_TYPE *>(indices.mutable_data()),
      static_cast<T *>(data.mutable_data()));
  return true;
}

template <typename T>
inline py::object make_shared_array_1d(const std::shared_ptr<SArray<T>> &value) {
  ensure_numpy_imported();
  npy_intp dims[1] = {static_cast<npy_intp>(value->size())};
  auto *array = reinterpret_cast<PyArrayObject *>(
      PyArray_SimpleNewFromData(1, dims, numpy_type<T>::value, value->data()));
  if (array == nullptr) throw py::error_already_set();

  if (value->data_owner() != nullptr) {
    PyObject *base = reinterpret_cast<PyObject *>(value->data_owner());
    Py_INCREF(base);
    if (PyArray_SetBaseObject(array, base) < 0) {
      Py_DECREF(base);
      Py_DECREF(reinterpret_cast<PyObject *>(array));
      throw py::error_already_set();
    }
  } else {
    PyArray_ENABLEFLAGS(array, NPY_ARRAY_OWNDATA);
    value->give_data_ownership(array);
  }

  return py::reinterpret_steal<py::object>(reinterpret_cast<PyObject *>(array));
}

template <typename T, typename MAJ = RowMajor>
inline py::object make_shared_array_2d(const std::shared_ptr<SArray2d<T, MAJ>> &value) {
  ensure_numpy_imported();
  npy_intp dims[2] = {static_cast<npy_intp>(value->n_rows()),
                      static_cast<npy_intp>(value->n_cols())};
  PyArrayObject *array = nullptr;
  if constexpr (major_traits<MAJ>::fortran) {
    array = reinterpret_cast<PyArrayObject *>(PyArray_New(
        &PyArray_Type, 2, dims, numpy_type<T>::value, nullptr, value->data(), 0,
        NPY_ARRAY_F_CONTIGUOUS, nullptr));
  } else {
    array = reinterpret_cast<PyArrayObject *>(
        PyArray_SimpleNewFromData(2, dims, numpy_type<T>::value, value->data()));
  }
  if (array == nullptr) throw py::error_already_set();

  if (value->data_owner() != nullptr) {
    PyObject *base = reinterpret_cast<PyObject *>(value->data_owner());
    Py_INCREF(base);
    if (PyArray_SetBaseObject(array, base) < 0) {
      Py_DECREF(base);
      Py_DECREF(reinterpret_cast<PyObject *>(array));
      throw py::error_already_set();
    }
  } else {
    PyArray_ENABLEFLAGS(array, NPY_ARRAY_OWNDATA);
    value->give_data_ownership(array);
  }

  return py::reinterpret_steal<py::object>(reinterpret_cast<PyObject *>(array));
}

template <typename T, typename MAJ = RowMajor>
inline py::object make_dense_array_2d_copy(const Array2d<T, MAJ> &value) {
  const py::ssize_t n_rows = static_cast<py::ssize_t>(value.n_rows());
  const py::ssize_t n_cols = static_cast<py::ssize_t>(value.n_cols());
  py::array result(
      py::dtype::of<T>(), {n_rows, n_cols},
      major_traits<MAJ>::fortran
          ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(sizeof(T)),
                                     static_cast<py::ssize_t>(sizeof(T) * n_rows)}
          : std::vector<py::ssize_t>{static_cast<py::ssize_t>(sizeof(T) * n_cols),
                                     static_cast<py::ssize_t>(sizeof(T))});
  auto *buffer = static_cast<T *>(result.mutable_data());
  std::copy(value.data(), value.data() + value.size(), buffer);
  return result;
}

template <typename T>
inline py::object make_dense_array_1d_copy(const Array<T> &value) {
  py::array_t<T> result({static_cast<py::ssize_t>(value.size())});
  auto *buffer = static_cast<T *>(result.mutable_data());
  std::copy(value.data(), value.data() + value.size(), buffer);
  return result;
}

template <typename T>
inline py::object make_sparse_array_1d(const std::shared_ptr<SSparseArray<T>> &value) {
  py::module_ scipy_sparse = py::module_::import("scipy.sparse");
  py::object data;
  py::object indices;
  py::array_t<INDICE_TYPE> indptr(2);
  auto *indptr_buffer = static_cast<INDICE_TYPE *>(indptr.mutable_data());
  indptr_buffer[0] = 0;
  indptr_buffer[1] = static_cast<INDICE_TYPE>(value->size_sparse());
  const bool has_data_owner = value->data_owner() != nullptr;
  const bool has_indices_owner = value->indices_owner() != nullptr;
  const bool has_any_owner = has_data_owner || has_indices_owner;
  const bool has_all_owners = has_data_owner && has_indices_owner;

  if (has_all_owners) {
    data = py::reinterpret_borrow<py::object>(
        reinterpret_cast<PyObject *>(value->data_owner()));
    indices = py::reinterpret_borrow<py::object>(
        reinterpret_cast<PyObject *>(value->indices_owner()));
  } else if (!has_any_owner) {
    ensure_numpy_imported();
    npy_intp dims[1] = {static_cast<npy_intp>(value->size_sparse())};

    auto *data_array = reinterpret_cast<PyArrayObject *>(
        PyArray_SimpleNewFromData(1, dims, numpy_type<T>::value, value->data()));
    if (data_array == nullptr) throw py::error_already_set();
    PyArray_ENABLEFLAGS(data_array, NPY_ARRAY_OWNDATA);
    data = py::reinterpret_steal<py::object>(reinterpret_cast<PyObject *>(data_array));

    auto *indices_array = reinterpret_cast<PyArrayObject *>(PyArray_SimpleNewFromData(
        1, dims, numpy_type<INDICE_TYPE>::value, value->indices()));
    if (indices_array == nullptr) throw py::error_already_set();
    PyArray_ENABLEFLAGS(indices_array, NPY_ARRAY_OWNDATA);
    indices =
        py::reinterpret_steal<py::object>(reinterpret_cast<PyObject *>(indices_array));

    value->give_data_indices_owners(data.ptr(), indices.ptr());
  } else {
    py::array_t<T> data_copy(static_cast<py::ssize_t>(value->size_sparse()));
    std::copy(value->data(), value->data() + value->size_sparse(),
              static_cast<T *>(data_copy.mutable_data()));
    data = std::move(data_copy);

    py::array_t<INDICE_TYPE> indices_copy(static_cast<py::ssize_t>(value->size_sparse()));
    std::copy(value->indices(), value->indices() + value->size_sparse(),
              static_cast<INDICE_TYPE *>(indices_copy.mutable_data()));
    indices = std::move(indices_copy);
  }

  py::tuple shape = py::make_tuple(1, value->size());
  return scipy_sparse.attr("csr_matrix")(py::make_tuple(data, indices, indptr),
                                         py::arg("shape") = shape,
                                         py::arg("copy") = false);
}

template <typename T, typename MAJ = RowMajor>
inline py::object make_sparse_array_2d(const std::shared_ptr<SSparseArray2d<T, MAJ>> &value) {
  py::module_ scipy_sparse = py::module_::import("scipy.sparse");
  py::object data;
  py::object indices;
  py::object indptr;
  const bool has_data_owner = value->data_owner() != nullptr;
  const bool has_indices_owner = value->indices_owner() != nullptr;
  const bool has_indptr_owner = value->row_indices_owner() != nullptr;
  const bool has_any_owner =
      has_data_owner || has_indices_owner || has_indptr_owner;
  const bool has_all_owners =
      has_data_owner && has_indices_owner && has_indptr_owner;

  if (has_all_owners) {
    data = py::reinterpret_borrow<py::object>(
        reinterpret_cast<PyObject *>(value->data_owner()));
    indices = py::reinterpret_borrow<py::object>(
        reinterpret_cast<PyObject *>(value->indices_owner()));
    indptr = py::reinterpret_borrow<py::object>(
        reinterpret_cast<PyObject *>(value->row_indices_owner()));
  } else if (!has_any_owner) {
    ensure_numpy_imported();
    npy_intp sparse_dims[1] = {static_cast<npy_intp>(value->size_sparse())};
    npy_intp indptr_dims[1] = {static_cast<npy_intp>(value->n_rows() + 1)};

    auto *data_array = reinterpret_cast<PyArrayObject *>(
        PyArray_SimpleNewFromData(1, sparse_dims, numpy_type<T>::value, value->data()));
    if (data_array == nullptr) throw py::error_already_set();
    PyArray_ENABLEFLAGS(data_array, NPY_ARRAY_OWNDATA);
    data = py::reinterpret_steal<py::object>(reinterpret_cast<PyObject *>(data_array));

    auto *indices_array = reinterpret_cast<PyArrayObject *>(PyArray_SimpleNewFromData(
        1, sparse_dims, numpy_type<INDICE_TYPE>::value, value->indices()));
    if (indices_array == nullptr) throw py::error_already_set();
    PyArray_ENABLEFLAGS(indices_array, NPY_ARRAY_OWNDATA);
    indices =
        py::reinterpret_steal<py::object>(reinterpret_cast<PyObject *>(indices_array));

    auto *indptr_array = reinterpret_cast<PyArrayObject *>(PyArray_SimpleNewFromData(
        1, indptr_dims, numpy_type<INDICE_TYPE>::value, value->row_indices()));
    if (indptr_array == nullptr) throw py::error_already_set();
    PyArray_ENABLEFLAGS(indptr_array, NPY_ARRAY_OWNDATA);
    indptr =
        py::reinterpret_steal<py::object>(reinterpret_cast<PyObject *>(indptr_array));

    value->give_data_indices_rowindices_owners(data.ptr(), indices.ptr(),
                                               indptr.ptr());
  } else {
    py::array_t<T> data_copy(static_cast<py::ssize_t>(value->size_sparse()));
    std::copy(value->data(), value->data() + value->size_sparse(),
              static_cast<T *>(data_copy.mutable_data()));
    data = std::move(data_copy);

    py::array_t<INDICE_TYPE> indices_copy(static_cast<py::ssize_t>(value->size_sparse()));
    std::copy(value->indices(), value->indices() + value->size_sparse(),
              static_cast<INDICE_TYPE *>(indices_copy.mutable_data()));
    indices = std::move(indices_copy);

    py::array_t<INDICE_TYPE> indptr_copy(static_cast<py::ssize_t>(value->n_rows() + 1));
    std::copy(value->row_indices(), value->row_indices() + value->n_rows() + 1,
              static_cast<INDICE_TYPE *>(indptr_copy.mutable_data()));
    indptr = std::move(indptr_copy);
  }

  py::tuple shape = py::make_tuple(value->n_rows(), value->n_cols());
  py::str ctor = major_traits<MAJ>::fortran ? "csc_matrix" : "csr_matrix";
  return scipy_sparse.attr(ctor)(py::make_tuple(data, indices, indptr),
                                 py::arg("shape") = shape,
                                 py::arg("copy") = false);
}

template <typename T>
inline py::object cast_base_array_copy(const BaseArray<T> &value) {
  if (value.is_dense()) {
    py::array_t<T> result({static_cast<py::ssize_t>(value.size())});
    auto *buffer = static_cast<T *>(result.mutable_data());
    std::copy(value.data(), value.data() + value.size(), buffer);
    return result;
  }
  SparseArray<T> tmp(value.size(), value.size_sparse(), value.indices(),
                     value.data());
  auto ptr = SSparseArray<T>::new_ptr(tmp);
  return make_sparse_array_1d(ptr);
}

template <typename T, typename MAJ = RowMajor>
inline py::object cast_base_array2d_copy(const BaseArray2d<T, MAJ> &value) {
  if (value.is_dense()) {
    const py::ssize_t n_rows = static_cast<py::ssize_t>(value.n_rows());
    const py::ssize_t n_cols = static_cast<py::ssize_t>(value.n_cols());
    py::array result(
        py::dtype::of<T>(), {n_rows, n_cols},
        major_traits<MAJ>::fortran
            ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(sizeof(T)),
                                       static_cast<py::ssize_t>(sizeof(T) * n_rows)}
            : std::vector<py::ssize_t>{static_cast<py::ssize_t>(sizeof(T) * n_cols),
                                       static_cast<py::ssize_t>(sizeof(T))});
    auto *buffer = static_cast<T *>(result.mutable_data());
    std::copy(value.data(), value.data() + value.size(), buffer);
    return result;
  }
  SparseArray2d<T, MAJ> tmp(value.n_rows(), value.n_cols(), value.row_indices(),
                            value.indices(), value.data());
  auto ptr = SSparseArray2d<T, MAJ>::new_ptr(tmp);
  return make_sparse_array_2d(ptr);
}

}  // namespace tick::pybind

namespace pybind11::detail {

template <typename T>
struct type_caster<Array<T>> {
 public:
  PYBIND11_TYPE_CASTER(Array<T>, _("Array"));

  bool load(handle src, bool) {
    return tick::pybind::load_dense_array_1d<T>(src, value, owner);
  }

  static handle cast(const Array<T> &src, return_value_policy, handle) {
    return tick::pybind::make_dense_array_1d_copy(src).release().ptr();
  }

 private:
  py::object owner;
};

template <typename T, typename MAJ>
struct type_caster<Array2d<T, MAJ>> {
 public:
  using type = Array2d<T, MAJ>;
  PYBIND11_TYPE_CASTER(type, _("Array2d"));

  bool load(handle src, bool) {
    return tick::pybind::load_dense_array_2d<T, MAJ>(src, value, owner);
  }

  static handle cast(const Array2d<T, MAJ> &src, return_value_policy, handle) {
    return tick::pybind::make_dense_array_2d_copy(src).release().ptr();
  }

 private:
  py::object owner;
};

template <typename T>
struct type_caster<SparseArray<T>> {
 public:
  PYBIND11_TYPE_CASTER(SparseArray<T>, _("SparseArray"));

  bool load(handle src, bool) {
    return tick::pybind::load_sparse_array_1d<T>(src, value, data_owner,
                                                 indices_owner);
  }

 private:
  py::object data_owner;
  py::object indices_owner;
};

template <typename T, typename MAJ>
struct type_caster<SparseArray2d<T, MAJ>> {
 public:
  using type = SparseArray2d<T, MAJ>;
  PYBIND11_TYPE_CASTER(type, _("SparseArray2d"));

  bool load(handle src, bool) {
    return tick::pybind::load_sparse_array_2d<T, MAJ>(src, value, data_owner,
                                                      indices_owner,
                                                      indptr_owner);
  }

 private:
  py::object data_owner;
  py::object indices_owner;
  py::object indptr_owner;
};

template <typename T>
struct type_caster<BaseArray<T>> {
 public:
  PYBIND11_TYPE_CASTER(BaseArray<T>, _("BaseArray"));

  bool load(handle src, bool) {
    Array<T> dense;
    if (tick::pybind::load_dense_array_1d<T>(src, dense, owner)) {
      value = std::move(static_cast<BaseArray<T> &>(dense));
      return true;
    }

    SparseArray<T> sparse;
    return tick::pybind::load_sparse_array_1d<T>(src, sparse, data_owner,
                                                 indices_owner)
               ? (value = std::move(static_cast<BaseArray<T> &>(sparse)), true)
               : false;
  }

  static handle cast(const BaseArray<T> &src, return_value_policy, handle) {
    return tick::pybind::cast_base_array_copy(src).release().ptr();
  }

 private:
  py::object owner;
  py::object data_owner;
  py::object indices_owner;
};

template <typename T, typename MAJ>
struct type_caster<BaseArray2d<T, MAJ>> {
 public:
  using type = BaseArray2d<T, MAJ>;
  PYBIND11_TYPE_CASTER(type, _("BaseArray2d"));

  bool load(handle src, bool) {
    Array2d<T, MAJ> dense;
    if (tick::pybind::load_dense_array_2d<T, MAJ>(src, dense, owner)) {
      value = std::move(static_cast<BaseArray2d<T, MAJ> &>(dense));
      return true;
    }

    SparseArray2d<T, MAJ> sparse;
    return tick::pybind::load_sparse_array_2d<T, MAJ>(src, sparse, data_owner,
                                                      indices_owner, indptr_owner)
               ? (value = std::move(static_cast<BaseArray2d<T, MAJ> &>(sparse)),
                  true)
               : false;
  }

  static handle cast(const BaseArray2d<T, MAJ> &src, return_value_policy, handle) {
    return tick::pybind::cast_base_array2d_copy(src).release().ptr();
  }

 private:
  py::object owner;
  py::object data_owner;
  py::object indices_owner;
  py::object indptr_owner;
};

template <typename T>
struct type_caster<std::shared_ptr<SArray<T>>> {
 public:
  PYBIND11_TYPE_CASTER(std::shared_ptr<SArray<T>>, _("SArrayPtr"));

  bool load(handle src, bool) {
    if (!py::isinstance<py::array>(src)) return false;
    py::array array = py::reinterpret_borrow<py::array>(src);
    if (array.ndim() != 1 || !tick::pybind::dtype_matches<T>(array)) return false;
    auto *array_obj = reinterpret_cast<PyArrayObject *>(array.ptr());
    if (!PyArray_ISCARRAY(array_obj) || !PyArray_ISWRITEABLE(array_obj)) return false;
    value = SArray<T>::new_ptr();
    value->set_data(static_cast<T *>(PyArray_DATA(array_obj)),
                    static_cast<ulong>(array.shape(0)), array.ptr());
    owner = py::reinterpret_borrow<py::object>(src);
    return true;
  }

  static handle cast(const std::shared_ptr<SArray<T>> &src, return_value_policy, handle) {
    if (!src) return py::none().release().ptr();
    return tick::pybind::make_shared_array_1d(src).release().ptr();
  }

 private:
  py::object owner;
};

template <typename T, typename MAJ>
struct type_caster<std::shared_ptr<SArray2d<T, MAJ>>> {
 public:
  using type = std::shared_ptr<SArray2d<T, MAJ>>;
  PYBIND11_TYPE_CASTER(type, _("SArray2dPtr"));

  bool load(handle src, bool) {
    if (!py::isinstance<py::array>(src)) return false;
    py::array array = py::reinterpret_borrow<py::array>(src);
    if (array.ndim() != 2 || !tick::pybind::dtype_matches<T>(array)) return false;
    auto *array_obj = reinterpret_cast<PyArrayObject *>(array.ptr());
    const bool contiguous = tick::pybind::major_traits<MAJ>::fortran
                                ? (PyArray_IS_F_CONTIGUOUS(array_obj) &&
                                   PyArray_ISWRITEABLE(array_obj))
                                : (PyArray_IS_C_CONTIGUOUS(array_obj) &&
                                   PyArray_ISWRITEABLE(array_obj));
    if (!contiguous) return false;
    value = SArray2d<T, MAJ>::new_ptr();
    value->set_data(static_cast<T *>(PyArray_DATA(array_obj)),
                    static_cast<ulong>(array.shape(0)),
                    static_cast<ulong>(array.shape(1)), array.ptr());
    owner = py::reinterpret_borrow<py::object>(src);
    return true;
  }

  static handle cast(const std::shared_ptr<SArray2d<T, MAJ>> &src,
                     return_value_policy, handle) {
    if (!src) return py::none().release().ptr();
    return tick::pybind::make_shared_array_2d(src).release().ptr();
  }

 private:
  py::object owner;
};

template <typename T>
struct type_caster<std::shared_ptr<VArray<T>>> {
 public:
  PYBIND11_TYPE_CASTER(std::shared_ptr<VArray<T>>, _("VArrayPtr"));

  bool load(handle src, bool) {
    if (!py::isinstance<py::array>(src)) return false;
    py::array array = py::reinterpret_borrow<py::array>(src);
    if (array.ndim() != 1 || !tick::pybind::dtype_matches<T>(array)) return false;
    auto *array_obj = reinterpret_cast<PyArrayObject *>(array.ptr());
    if (!PyArray_ISCARRAY(array_obj) || !PyArray_ISWRITEABLE(array_obj)) return false;
    value = VArray<T>::new_ptr();
    value->set_data(static_cast<T *>(PyArray_DATA(array_obj)),
                    static_cast<ulong>(array.shape(0)), array.ptr());
    owner = py::reinterpret_borrow<py::object>(src);
    return true;
  }

  static handle cast(const std::shared_ptr<VArray<T>> &src, return_value_policy, handle) {
    if (!src) return py::none().release().ptr();
    return tick::pybind::make_shared_array_1d(
               std::static_pointer_cast<SArray<T>>(src))
        .release()
        .ptr();
  }

 private:
  py::object owner;
};

template <typename T>
struct type_caster<std::shared_ptr<SSparseArray<T>>> {
 public:
  PYBIND11_TYPE_CASTER(std::shared_ptr<SSparseArray<T>>, _("SSparseArrayPtr"));

  bool load(handle src, bool) {
    SparseArray<T> sparse;
    if (!tick::pybind::load_sparse_array_1d<T>(src, sparse, data_owner,
                                               indices_owner)) {
      return false;
    }
    value = SSparseArray<T>::new_ptr(0, 0);
    value->set_data_indices(sparse.data(), sparse.indices(), sparse.size(),
                            sparse.size_sparse(), data_owner.ptr(),
                            indices_owner.ptr());
    return true;
  }

  static handle cast(const std::shared_ptr<SSparseArray<T>> &src,
                     return_value_policy, handle) {
    if (!src) return py::none().release().ptr();
    return tick::pybind::make_sparse_array_1d(src).release().ptr();
  }

 private:
  py::object data_owner;
  py::object indices_owner;
};

template <typename T, typename MAJ>
struct type_caster<std::shared_ptr<SSparseArray2d<T, MAJ>>> {
 public:
  using type = std::shared_ptr<SSparseArray2d<T, MAJ>>;
  PYBIND11_TYPE_CASTER(type, _("SSparseArray2dPtr"));

  bool load(handle src, bool) {
    SparseArray2d<T, MAJ> sparse;
    if (!tick::pybind::load_sparse_array_2d<T, MAJ>(src, sparse, data_owner,
                                                    indices_owner,
                                                    indptr_owner)) {
      return false;
    }
    value = SSparseArray2d<T, MAJ>::new_ptr(0, 0, 0);
    value->set_data_indices_rowindices(
        sparse.data(), sparse.indices(), sparse.row_indices(), sparse.n_rows(),
        sparse.n_cols(), data_owner.ptr(), indices_owner.ptr(),
        indptr_owner.ptr());
    return true;
  }

  static handle cast(const std::shared_ptr<SSparseArray2d<T, MAJ>> &src,
                     return_value_policy, handle) {
    if (!src) return py::none().release().ptr();
    return tick::pybind::make_sparse_array_2d(src).release().ptr();
  }

 private:
  py::object data_owner;
  py::object indices_owner;
  py::object indptr_owner;
};

template <typename T>
struct type_caster<std::shared_ptr<BaseArray<T>>> {
 public:
  PYBIND11_TYPE_CASTER(std::shared_ptr<BaseArray<T>>, _("SBaseArrayPtr"));

  bool load(handle src, bool) {
    if (py::isinstance<py::array>(src)) {
      py::array array = py::reinterpret_borrow<py::array>(src);
      if (array.ndim() != 1 || !tick::pybind::dtype_matches<T>(array)) return false;
      auto *array_obj = reinterpret_cast<PyArrayObject *>(array.ptr());
      if (!PyArray_ISCARRAY(array_obj) || !PyArray_ISWRITEABLE(array_obj)) return false;
      auto dense = SArray<T>::new_ptr();
      dense->set_data(static_cast<T *>(PyArray_DATA(array_obj)),
                      static_cast<ulong>(array.shape(0)), array.ptr());
      owner = py::reinterpret_borrow<py::object>(src);
      value = dense;
      return true;
    }

    SparseArray<T> sparse;
    if (!tick::pybind::load_sparse_array_1d<T>(src, sparse, data_owner,
                                               indices_owner)) {
      return false;
    }
    auto shared = SSparseArray<T>::new_ptr(0, 0);
    shared->set_data_indices(sparse.data(), sparse.indices(), sparse.size(),
                             sparse.size_sparse(), data_owner.ptr(),
                             indices_owner.ptr());
    value = shared;
    return true;
  }

  static handle cast(const std::shared_ptr<BaseArray<T>> &src,
                     return_value_policy, handle) {
    if (!src) return py::none().release().ptr();
    if (src->is_dense()) {
      auto dense = std::dynamic_pointer_cast<SArray<T>>(src);
      if (dense) return tick::pybind::make_shared_array_1d(dense).release().ptr();
    } else {
      auto sparse = std::dynamic_pointer_cast<SSparseArray<T>>(src);
      if (sparse) return tick::pybind::make_sparse_array_1d(sparse).release().ptr();
    }
    return tick::pybind::cast_base_array_copy(*src).release().ptr();
  }

 private:
  py::object owner;
  py::object data_owner;
  py::object indices_owner;
};

template <typename T, typename MAJ>
struct type_caster<std::shared_ptr<BaseArray2d<T, MAJ>>> {
 public:
  using type = std::shared_ptr<BaseArray2d<T, MAJ>>;
  PYBIND11_TYPE_CASTER(type, _("SBaseArray2dPtr"));

  bool load(handle src, bool) {
    if (py::isinstance<py::array>(src)) {
      py::array array = py::reinterpret_borrow<py::array>(src);
      if (array.ndim() != 2 || !tick::pybind::dtype_matches<T>(array)) return false;
      auto *array_obj = reinterpret_cast<PyArrayObject *>(array.ptr());
      const bool contiguous = tick::pybind::major_traits<MAJ>::fortran
                                  ? (PyArray_IS_F_CONTIGUOUS(array_obj) &&
                                     PyArray_ISWRITEABLE(array_obj))
                                  : (PyArray_IS_C_CONTIGUOUS(array_obj) &&
                                     PyArray_ISWRITEABLE(array_obj));
      if (!contiguous) return false;
      auto dense = SArray2d<T, MAJ>::new_ptr();
      dense->set_data(static_cast<T *>(PyArray_DATA(array_obj)),
                      static_cast<ulong>(array.shape(0)),
                      static_cast<ulong>(array.shape(1)), array.ptr());
      owner = py::reinterpret_borrow<py::object>(src);
      value = dense;
      return true;
    }

    SparseArray2d<T, MAJ> sparse;
    if (!tick::pybind::load_sparse_array_2d<T, MAJ>(src, sparse, data_owner,
                                                    indices_owner,
                                                    indptr_owner)) {
      return false;
    }
    auto shared = SSparseArray2d<T, MAJ>::new_ptr(0, 0, 0);
    shared->set_data_indices_rowindices(
        sparse.data(), sparse.indices(), sparse.row_indices(), sparse.n_rows(),
        sparse.n_cols(), data_owner.ptr(), indices_owner.ptr(),
        indptr_owner.ptr());
    value = shared;
    return true;
  }

  static handle cast(const std::shared_ptr<BaseArray2d<T, MAJ>> &src,
                     return_value_policy, handle) {
    if (!src) return py::none().release().ptr();
    if (src->is_dense()) {
      auto dense = std::dynamic_pointer_cast<SArray2d<T, MAJ>>(src);
      if (dense) return tick::pybind::make_shared_array_2d(dense).release().ptr();
    } else {
      auto sparse = std::dynamic_pointer_cast<SSparseArray2d<T, MAJ>>(src);
      if (sparse) return tick::pybind::make_sparse_array_2d(sparse).release().ptr();
    }
    return tick::pybind::cast_base_array2d_copy(*src).release().ptr();
  }

 private:
  py::object owner;
  py::object data_owner;
  py::object indices_owner;
  py::object indptr_owner;
};

}  // namespace pybind11::detail
