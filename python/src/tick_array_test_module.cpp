#include <cstdint>
#include <string>

#include <pybind11/pybind11.h>

#include "common/tick_pybind11_arrays.h"
#include "tick/array_test/array_test.h"
#include "tick/array_test/performance_test.h"
#include "tick/array_test/sbasearray_container.h"
#include "tick/array_test/typemap_test.h"
#include "tick/array_test/varraycontainer.h"

namespace py = pybind11;

namespace {

template <typename T>
const char *cpp_type_label();

template <>
const char *cpp_type_label<double>() {
  return "double";
}

template <>
const char *cpp_type_label<float>() {
  return "float";
}

template <>
const char *cpp_type_label<std::int32_t>() {
  return "std::int32_t";
}

template <>
const char *cpp_type_label<std::uint32_t>() {
  return "std::uint32_t";
}

template <>
const char *cpp_type_label<std::int16_t>() {
  return "std::int16_t";
}

template <>
const char *cpp_type_label<std::uint16_t>() {
  return "std::uint16_t";
}

template <>
const char *cpp_type_label<std::int64_t>() {
  return "std::int64_t";
}

template <>
const char *cpp_type_label<std::uint64_t>() {
  return "std::uint64_t";
}

[[noreturn]] void raise_value_error(const std::string &message) {
  PyErr_SetString(PyExc_ValueError, message.c_str());
  throw py::error_already_set();
}

template <typename T>
void validate_dense_array_1d(const py::handle &src) {
  if (tick::pybind::is_scipy_sparse(src)) {
    raise_value_error("Expecting a dense numpy array");
  }
  if (!py::isinstance<py::array>(src)) {
    raise_value_error("Expecting a dense 1d " +
                      std::string(cpp_type_label<T>()) + " numpy array");
  }

  py::array array = py::reinterpret_borrow<py::array>(src);
  if (array.ndim() != 1) {
    raise_value_error("Expecting a 1 dimensional contiguous numpy array");
  }
  auto *array_obj = reinterpret_cast<PyArrayObject *>(array.ptr());
  if (!PyArray_ISCARRAY(array_obj) || !PyArray_ISWRITEABLE(array_obj)) {
    raise_value_error("Expecting a contiguous writable dense numpy array");
  }
  if (!tick::pybind::dtype_matches<T>(array)) {
    raise_value_error("Expecting a " + std::string(cpp_type_label<T>()) +
                      " numpy array");
  }
}

template <typename T>
void validate_dense_array_2d(const py::handle &src) {
  if (tick::pybind::is_scipy_sparse(src)) {
    raise_value_error("Expecting a dense numpy array");
  }
  if (!py::isinstance<py::array>(src)) {
    raise_value_error("Expecting a dense 2d " +
                      std::string(cpp_type_label<T>()) + " numpy array");
  }

  py::array array = py::reinterpret_borrow<py::array>(src);
  if (array.ndim() != 2) {
    raise_value_error("Expecting a 2 dimensional contiguous numpy array");
  }
  auto *array_obj = reinterpret_cast<PyArrayObject *>(array.ptr());
  if (!PyArray_IS_C_CONTIGUOUS(array_obj) || !PyArray_ISWRITEABLE(array_obj)) {
    raise_value_error("Expecting a contiguous writable 2 dimensional numpy array");
  }
  if (!tick::pybind::dtype_matches<T>(array)) {
    raise_value_error("Expecting a " + std::string(cpp_type_label<T>()) +
                      " numpy array");
  }
}

template <typename T>
void validate_sparse_array_1d(const py::handle &src) {
  if (py::isinstance<py::array>(src)) {
    raise_value_error("Expecting a sparse array");
  }
  if (!tick::pybind::is_scipy_sparse(src)) {
    raise_value_error("Expecting a sparse array");
  }

  py::tuple shape = py::reinterpret_borrow<py::tuple>(src.attr("shape"));
  if (shape.size() != 2 || py::cast<py::ssize_t>(shape[0]) != 1) {
    raise_value_error("Expecting a 1d sparse array");
  }

  py::array data = py::reinterpret_borrow<py::array>(src.attr("data"));
  py::array indices = py::reinterpret_borrow<py::array>(src.attr("indices"));
  if (!tick::pybind::dtype_matches<T>(data) ||
      !tick::pybind::index_dtype_matches(indices)) {
    raise_value_error("Expecting a sparse " +
                      std::string(cpp_type_label<T>()) + " array");
  }
}

template <typename T>
void validate_base_array_1d(const py::handle &src) {
  if (py::isinstance<py::array>(src)) {
    validate_dense_array_1d<T>(src);
    return;
  }
  if (tick::pybind::is_scipy_sparse(src)) {
    validate_sparse_array_1d<T>(src);
    return;
  }

  raise_value_error("Expecting a 1d " + std::string(cpp_type_label<T>()) +
                    " numpy array or sparse array");
}

void validate_array_list_1d(const py::handle &src) {
  if (!py::isinstance<py::list>(src)) {
    raise_value_error("Expecting a list of numpy array objects");
  }
}

void validate_array_list_2d(const py::handle &src) {
  if (!py::isinstance<py::list>(src)) {
    raise_value_error("Expecting a 2d list of numpy array objects");
  }

  py::list outer = py::reinterpret_borrow<py::list>(src);
  for (py::handle item : outer) {
    if (!py::isinstance<py::list>(item)) {
      raise_value_error("Expecting a 2d list of numpy array objects");
    }
  }
}

template <typename T>
void bind_typemap_validators(py::module_ &m, const std::string &suffix) {
  const std::string array_name = "test_typemap_in_not_ol_Array" + suffix;
  m.def(array_name.c_str(), [](py::object obj) { validate_dense_array_1d<T>(obj); });

  const std::string array2d_name = "test_typemap_in_not_ol_Array" + suffix + "2d";
  m.def(array2d_name.c_str(),
        [](py::object obj) { validate_dense_array_2d<T>(obj); });

  const std::string sparse_name =
      "test_typemap_in_not_ol_SparseArray" + suffix;
  m.def(sparse_name.c_str(),
        [](py::object obj) { validate_sparse_array_1d<T>(obj); });

  const std::string base_name = "test_typemap_in_not_ol_BaseArray" + suffix;
  m.def(base_name.c_str(), [](py::object obj) { validate_base_array_1d<T>(obj); });

  const std::string list1d_name =
      "test_typemap_in_not_ol_Array" + suffix + "List1D";
  m.def(list1d_name.c_str(),
        [](py::object obj) { validate_array_list_1d(obj); });

  const std::string list2d_name =
      "test_typemap_in_not_ol_Array" + suffix + "List2D";
  m.def(list2d_name.c_str(),
        [](py::object obj) { validate_array_list_2d(obj); });
}

#define DEF_FN(name, sig) m.def(#name, static_cast<sig>(&::name))

void bind_array_method_tests(py::module_ &m) {
  DEF_FN(test_constructor_ArrayDouble, double (*)(ulong));
  DEF_FN(test_arange, SArrayDoublePtr (*)(std::int64_t, std::int64_t));
  DEF_FN(test_IndexError_ArrayDouble, double (*)(ArrayDouble &));
  DEF_FN(test_IndexError_cols_ArrayDouble2d, double (*)(ArrayDouble2d &));
  DEF_FN(test_constructor_SparseArrayDouble, double (*)());
  DEF_FN(test_constructor_SparseArrayDouble2d, double (*)());
  DEF_FN(test_BaseArray_empty_constructor, bool (*)(bool));

#define BIND_INIT_TO_ZERO(ARRAY_TYPE) \
  DEF_FN(test_init_to_zero_##ARRAY_TYPE, void (*)(ARRAY_TYPE &))
#define BIND_INIT_TO_ZERO_PTR(ARRAY_PTR_TYPE) \
  DEF_FN(test_init_to_zero_##ARRAY_PTR_TYPE, void (*)(ARRAY_PTR_TYPE))

  BIND_INIT_TO_ZERO(BaseArrayDouble);
  BIND_INIT_TO_ZERO(ArrayDouble);
  BIND_INIT_TO_ZERO(SparseArrayDouble);
  BIND_INIT_TO_ZERO(BaseArrayDouble2d);
  BIND_INIT_TO_ZERO(ArrayDouble2d);
  BIND_INIT_TO_ZERO(SparseArrayDouble2d);
  BIND_INIT_TO_ZERO_PTR(SBaseArrayDoublePtr);
  BIND_INIT_TO_ZERO_PTR(SArrayDoublePtr);
  BIND_INIT_TO_ZERO_PTR(VArrayDoublePtr);
  BIND_INIT_TO_ZERO_PTR(SSparseArrayDoublePtr);
  BIND_INIT_TO_ZERO_PTR(SBaseArrayDouble2dPtr);
  BIND_INIT_TO_ZERO_PTR(SArrayDouble2dPtr);
  BIND_INIT_TO_ZERO_PTR(SSparseArrayDouble2dPtr);

#define BIND_COPY(ARRAY_TYPE) DEF_FN(test_copy_##ARRAY_TYPE, void (*)(ARRAY_TYPE &))
#define BIND_COPY_PTR(ARRAY_PTR_TYPE) \
  DEF_FN(test_copy_##ARRAY_PTR_TYPE, void (*)(ARRAY_PTR_TYPE))

  BIND_COPY(BaseArrayDouble);
  BIND_COPY(ArrayDouble);
  BIND_COPY(SparseArrayDouble);
  BIND_COPY(BaseArrayDouble2d);
  BIND_COPY(ArrayDouble2d);
  BIND_COPY(SparseArrayDouble2d);
  BIND_COPY_PTR(SBaseArrayDoublePtr);
  BIND_COPY_PTR(SArrayDoublePtr);
  BIND_COPY_PTR(VArrayDoublePtr);
  BIND_COPY_PTR(SSparseArrayDoublePtr);
  BIND_COPY_PTR(SBaseArrayDouble2dPtr);
  BIND_COPY_PTR(SArrayDouble2dPtr);
  BIND_COPY_PTR(SSparseArrayDouble2dPtr);

#define BIND_MOVE(ARRAY_TYPE) DEF_FN(test_move_##ARRAY_TYPE, bool (*)(ARRAY_TYPE &))
  BIND_MOVE(BaseArrayDouble);
  BIND_MOVE(ArrayDouble);
  BIND_MOVE(SparseArrayDouble);
  BIND_MOVE(BaseArrayDouble2d);
  BIND_MOVE(ArrayDouble2d);
  BIND_MOVE(SparseArrayDouble2d);

#define BIND_VALUE(ARRAY_TYPE) \
  DEF_FN(test_value_##ARRAY_TYPE, double (*)(ARRAY_TYPE &, ulong))
#define BIND_VALUE_PTR(ARRAY_PTR_TYPE) \
  DEF_FN(test_value_##ARRAY_PTR_TYPE, double (*)(ARRAY_PTR_TYPE, ulong))

  BIND_VALUE(BaseArrayDouble);
  BIND_VALUE(ArrayDouble);
  BIND_VALUE(SparseArrayDouble);
  BIND_VALUE_PTR(SBaseArrayDoublePtr);
  BIND_VALUE_PTR(SArrayDoublePtr);
  BIND_VALUE_PTR(VArrayDoublePtr);
  BIND_VALUE_PTR(SSparseArrayDoublePtr);

#define BIND_LAST(ARRAY_TYPE) DEF_FN(test_last_##ARRAY_TYPE, double (*)(ARRAY_TYPE &))
#define BIND_LAST_PTR(ARRAY_PTR_TYPE) \
  DEF_FN(test_last_##ARRAY_PTR_TYPE, double (*)(ARRAY_PTR_TYPE))

  BIND_LAST(BaseArrayDouble);
  BIND_LAST(ArrayDouble);
  BIND_LAST(SparseArrayDouble);
  BIND_LAST_PTR(SBaseArrayDoublePtr);
  BIND_LAST_PTR(SArrayDoublePtr);
  BIND_LAST_PTR(VArrayDoublePtr);
  BIND_LAST_PTR(SSparseArrayDoublePtr);

#define BIND_DOT(ARRAY_TYPE1, ARRAY_TYPE2) \
  DEF_FN(test_dot_##ARRAY_TYPE1##_##ARRAY_TYPE2, \
         double (*)(ARRAY_TYPE1 &, ARRAY_TYPE2 &))
#define BIND_DOT_PTR_1(ARRAY_PTR_TYPE1, ARRAY_TYPE2) \
  DEF_FN(test_dot_##ARRAY_PTR_TYPE1##_##ARRAY_TYPE2, \
         double (*)(ARRAY_PTR_TYPE1, ARRAY_TYPE2 &))
#define BIND_DOT_PTR_2(ARRAY_TYPE1, ARRAY_PTR_TYPE2) \
  DEF_FN(test_dot_##ARRAY_TYPE1##_##ARRAY_PTR_TYPE2, \
         double (*)(ARRAY_TYPE1 &, ARRAY_PTR_TYPE2))
#define BIND_DOT_PTR_PTR(ARRAY_PTR_TYPE1, ARRAY_PTR_TYPE2) \
  DEF_FN(test_dot_##ARRAY_PTR_TYPE1##_##ARRAY_PTR_TYPE2, \
         double (*)(ARRAY_PTR_TYPE1, ARRAY_PTR_TYPE2))

  BIND_DOT(BaseArrayDouble, BaseArrayDouble);
  BIND_DOT(BaseArrayDouble, ArrayDouble);
  BIND_DOT(BaseArrayDouble, SparseArrayDouble);
  BIND_DOT(ArrayDouble, BaseArrayDouble);
  BIND_DOT(ArrayDouble, ArrayDouble);
  BIND_DOT(ArrayDouble, SparseArrayDouble);
  BIND_DOT(SparseArrayDouble, BaseArrayDouble);
  BIND_DOT(SparseArrayDouble, ArrayDouble);
  BIND_DOT(SparseArrayDouble, SparseArrayDouble);

  BIND_DOT_PTR_1(SBaseArrayDoublePtr, BaseArrayDouble);
  BIND_DOT_PTR_1(SBaseArrayDoublePtr, ArrayDouble);
  BIND_DOT_PTR_1(SBaseArrayDoublePtr, SparseArrayDouble);
  BIND_DOT_PTR_1(SArrayDoublePtr, BaseArrayDouble);
  BIND_DOT_PTR_1(SArrayDoublePtr, ArrayDouble);
  BIND_DOT_PTR_1(SArrayDoublePtr, SparseArrayDouble);
  BIND_DOT_PTR_1(VArrayDoublePtr, BaseArrayDouble);
  BIND_DOT_PTR_1(VArrayDoublePtr, ArrayDouble);
  BIND_DOT_PTR_1(VArrayDoublePtr, SparseArrayDouble);
  BIND_DOT_PTR_1(SSparseArrayDoublePtr, BaseArrayDouble);
  BIND_DOT_PTR_1(SSparseArrayDoublePtr, ArrayDouble);
  BIND_DOT_PTR_1(SSparseArrayDoublePtr, SparseArrayDouble);

  BIND_DOT_PTR_2(BaseArrayDouble, SBaseArrayDoublePtr);
  BIND_DOT_PTR_2(BaseArrayDouble, SArrayDoublePtr);
  BIND_DOT_PTR_2(BaseArrayDouble, VArrayDoublePtr);
  BIND_DOT_PTR_2(BaseArrayDouble, SSparseArrayDoublePtr);
  BIND_DOT_PTR_2(ArrayDouble, SBaseArrayDoublePtr);
  BIND_DOT_PTR_2(ArrayDouble, SArrayDoublePtr);
  BIND_DOT_PTR_2(ArrayDouble, VArrayDoublePtr);
  BIND_DOT_PTR_2(ArrayDouble, SSparseArrayDoublePtr);
  BIND_DOT_PTR_2(SparseArrayDouble, SBaseArrayDoublePtr);
  BIND_DOT_PTR_2(SparseArrayDouble, SArrayDoublePtr);
  BIND_DOT_PTR_2(SparseArrayDouble, VArrayDoublePtr);
  BIND_DOT_PTR_2(SparseArrayDouble, SSparseArrayDoublePtr);

  BIND_DOT_PTR_PTR(SBaseArrayDoublePtr, SBaseArrayDoublePtr);
  BIND_DOT_PTR_PTR(SBaseArrayDoublePtr, SArrayDoublePtr);
  BIND_DOT_PTR_PTR(SBaseArrayDoublePtr, VArrayDoublePtr);
  BIND_DOT_PTR_PTR(SBaseArrayDoublePtr, SSparseArrayDoublePtr);
  BIND_DOT_PTR_PTR(SArrayDoublePtr, SBaseArrayDoublePtr);
  BIND_DOT_PTR_PTR(SArrayDoublePtr, SArrayDoublePtr);
  BIND_DOT_PTR_PTR(SArrayDoublePtr, VArrayDoublePtr);
  BIND_DOT_PTR_PTR(SArrayDoublePtr, SSparseArrayDoublePtr);
  BIND_DOT_PTR_PTR(VArrayDoublePtr, SBaseArrayDoublePtr);
  BIND_DOT_PTR_PTR(VArrayDoublePtr, SArrayDoublePtr);
  BIND_DOT_PTR_PTR(VArrayDoublePtr, VArrayDoublePtr);
  BIND_DOT_PTR_PTR(VArrayDoublePtr, SSparseArrayDoublePtr);
  BIND_DOT_PTR_PTR(SSparseArrayDoublePtr, SBaseArrayDoublePtr);
  BIND_DOT_PTR_PTR(SSparseArrayDoublePtr, SArrayDoublePtr);
  BIND_DOT_PTR_PTR(SSparseArrayDoublePtr, VArrayDoublePtr);
  BIND_DOT_PTR_PTR(SSparseArrayDoublePtr, SSparseArrayDoublePtr);

  DEF_FN(test_as_array, double (*)(BaseArrayDouble &));
  DEF_FN(test_as_array2d, double (*)(BaseArrayDouble2d &));
  DEF_FN(test_new_ptr_SArrayDoublePtr, double (*)(ArrayDouble &));
  DEF_FN(test_new_ptr_VArrayDoublePtr, double (*)(ArrayDouble &));
  DEF_FN(test_new_ptr_SSparseArrayDoublePtr, double (*)(SparseArrayDouble &));
  DEF_FN(test_new_ptr_SArrayDouble2dPtr, double (*)(ArrayDouble2d &));
  DEF_FN(test_new_ptr_SSparseArrayDouble2dPtr,
         double (*)(SparseArrayDouble2d &));

#define BIND_VIEW(ARRAY_TYPE) \
  DEF_FN(test_view_##ARRAY_TYPE, \
         SArrayDoublePtr (*)(ARRAY_TYPE &, ARRAY_TYPE &, ARRAY_TYPE &))

  BIND_VIEW(BaseArrayDouble);
  BIND_VIEW(ArrayDouble);
  BIND_VIEW(SparseArrayDouble);
  BIND_VIEW(BaseArrayDouble2d);
  BIND_VIEW(ArrayDouble2d);
  BIND_VIEW(SparseArrayDouble2d);

  DEF_FN(test_slice_view1d, SArrayDoublePtrList1D (*)(ArrayDouble &, ulong, ulong));
  DEF_FN(test_row_view_BaseArrayDouble2d,
         SArrayDoublePtrList1D (*)(BaseArrayDouble2d &, ulong));
  DEF_FN(test_row_view_ArrayDouble2d,
         SArrayDoublePtrList1D (*)(ArrayDouble2d &, ulong));
  DEF_FN(test_row_view_SparseArrayDouble2d,
         SArrayDoublePtrList1D (*)(SparseArrayDouble2d &, ulong));

  DEF_FN(test_as_array_ptr_ArrayDouble, SArrayDoublePtr (*)(ArrayDouble &));
  DEF_FN(test_as_array_ptr_SparseArrayDouble,
         SSparseArrayDoublePtr (*)(SparseArrayDouble &));
  DEF_FN(test_as_array_ptr_ArrayDouble2d,
         SArrayDouble2dPtr (*)(ArrayDouble2d &));
  DEF_FN(test_as_array_ptr_SparseArrayDouble2d,
         SSparseArrayDouble2dPtr (*)(SparseArrayDouble2d &));

#define BIND_SUM(ARRAY_TYPE) DEF_FN(test_sum_##ARRAY_TYPE, double (*)(ARRAY_TYPE &))
#define BIND_SUM_PTR(ARRAY_PTR_TYPE) \
  DEF_FN(test_sum_##ARRAY_PTR_TYPE, double (*)(ARRAY_PTR_TYPE))
#define BIND_MIN(ARRAY_TYPE) DEF_FN(test_min_##ARRAY_TYPE, double (*)(ARRAY_TYPE &))
#define BIND_MIN_PTR(ARRAY_PTR_TYPE) \
  DEF_FN(test_min_##ARRAY_PTR_TYPE, double (*)(ARRAY_PTR_TYPE))
#define BIND_MAX(ARRAY_TYPE) DEF_FN(test_max_##ARRAY_TYPE, double (*)(ARRAY_TYPE &))
#define BIND_MAX_PTR(ARRAY_PTR_TYPE) \
  DEF_FN(test_max_##ARRAY_PTR_TYPE, double (*)(ARRAY_PTR_TYPE))
#define BIND_NORM_SQ(ARRAY_TYPE) \
  DEF_FN(test_norm_sq_##ARRAY_TYPE, double (*)(ARRAY_TYPE &))
#define BIND_NORM_SQ_PTR(ARRAY_PTR_TYPE) \
  DEF_FN(test_norm_sq_##ARRAY_PTR_TYPE, double (*)(ARRAY_PTR_TYPE))

  BIND_SUM(BaseArrayDouble);
  BIND_SUM(ArrayDouble);
  BIND_SUM(SparseArrayDouble);
  BIND_SUM(BaseArrayDouble2d);
  BIND_SUM(ArrayDouble2d);
  BIND_SUM(SparseArrayDouble2d);
  BIND_SUM_PTR(SBaseArrayDoublePtr);
  BIND_SUM_PTR(SArrayDoublePtr);
  BIND_SUM_PTR(VArrayDoublePtr);
  BIND_SUM_PTR(SSparseArrayDoublePtr);
  BIND_SUM_PTR(SBaseArrayDouble2dPtr);
  BIND_SUM_PTR(SArrayDouble2dPtr);
  BIND_SUM_PTR(SSparseArrayDouble2dPtr);

  BIND_MIN(BaseArrayDouble);
  BIND_MIN(ArrayDouble);
  BIND_MIN(SparseArrayDouble);
  BIND_MIN(BaseArrayDouble2d);
  BIND_MIN(ArrayDouble2d);
  BIND_MIN(SparseArrayDouble2d);
  BIND_MIN_PTR(SBaseArrayDoublePtr);
  BIND_MIN_PTR(SArrayDoublePtr);
  BIND_MIN_PTR(VArrayDoublePtr);
  BIND_MIN_PTR(SSparseArrayDoublePtr);
  BIND_MIN_PTR(SBaseArrayDouble2dPtr);
  BIND_MIN_PTR(SArrayDouble2dPtr);
  BIND_MIN_PTR(SSparseArrayDouble2dPtr);

  BIND_MAX(BaseArrayDouble);
  BIND_MAX(ArrayDouble);
  BIND_MAX(SparseArrayDouble);
  BIND_MAX(BaseArrayDouble2d);
  BIND_MAX(ArrayDouble2d);
  BIND_MAX(SparseArrayDouble2d);
  BIND_MAX_PTR(SBaseArrayDoublePtr);
  BIND_MAX_PTR(SArrayDoublePtr);
  BIND_MAX_PTR(VArrayDoublePtr);
  BIND_MAX_PTR(SSparseArrayDoublePtr);
  BIND_MAX_PTR(SBaseArrayDouble2dPtr);
  BIND_MAX_PTR(SArrayDouble2dPtr);
  BIND_MAX_PTR(SSparseArrayDouble2dPtr);

  BIND_NORM_SQ(BaseArrayDouble);
  BIND_NORM_SQ(ArrayDouble);
  BIND_NORM_SQ(SparseArrayDouble);
  BIND_NORM_SQ(BaseArrayDouble2d);
  BIND_NORM_SQ(ArrayDouble2d);
  BIND_NORM_SQ(SparseArrayDouble2d);
  BIND_NORM_SQ_PTR(SBaseArrayDoublePtr);
  BIND_NORM_SQ_PTR(SArrayDoublePtr);
  BIND_NORM_SQ_PTR(VArrayDoublePtr);
  BIND_NORM_SQ_PTR(SSparseArrayDoublePtr);
  BIND_NORM_SQ_PTR(SBaseArrayDouble2dPtr);
  BIND_NORM_SQ_PTR(SArrayDouble2dPtr);
  BIND_NORM_SQ_PTR(SSparseArrayDouble2dPtr);

#define BIND_MULTIPLY(ARRAY_TYPE) \
  DEF_FN(test_multiply_##ARRAY_TYPE, void (*)(ARRAY_TYPE &, double))
#define BIND_MULTIPLY_PTR(ARRAY_PTR_TYPE) \
  DEF_FN(test_multiply_##ARRAY_PTR_TYPE, void (*)(ARRAY_PTR_TYPE, double))
#define BIND_DIVIDE(ARRAY_TYPE) \
  DEF_FN(test_divide_##ARRAY_TYPE, void (*)(ARRAY_TYPE &, double))
#define BIND_DIVIDE_PTR(ARRAY_PTR_TYPE) \
  DEF_FN(test_divide_##ARRAY_PTR_TYPE, void (*)(ARRAY_PTR_TYPE, double))

  BIND_MULTIPLY(BaseArrayDouble);
  BIND_MULTIPLY(ArrayDouble);
  BIND_MULTIPLY(SparseArrayDouble);
  BIND_MULTIPLY(BaseArrayDouble2d);
  BIND_MULTIPLY(ArrayDouble2d);
  BIND_MULTIPLY(SparseArrayDouble2d);
  BIND_MULTIPLY_PTR(SBaseArrayDoublePtr);
  BIND_MULTIPLY_PTR(SArrayDoublePtr);
  BIND_MULTIPLY_PTR(VArrayDoublePtr);
  BIND_MULTIPLY_PTR(SSparseArrayDoublePtr);
  BIND_MULTIPLY_PTR(SBaseArrayDouble2dPtr);
  BIND_MULTIPLY_PTR(SArrayDouble2dPtr);
  BIND_MULTIPLY_PTR(SSparseArrayDouble2dPtr);

  BIND_DIVIDE(BaseArrayDouble);
  BIND_DIVIDE(ArrayDouble);
  BIND_DIVIDE(SparseArrayDouble);
  BIND_DIVIDE(BaseArrayDouble2d);
  BIND_DIVIDE(ArrayDouble2d);
  BIND_DIVIDE(SparseArrayDouble2d);
  BIND_DIVIDE_PTR(SBaseArrayDoublePtr);
  BIND_DIVIDE_PTR(SArrayDoublePtr);
  BIND_DIVIDE_PTR(VArrayDoublePtr);
  BIND_DIVIDE_PTR(SSparseArrayDoublePtr);
  BIND_DIVIDE_PTR(SBaseArrayDouble2dPtr);
  BIND_DIVIDE_PTR(SArrayDouble2dPtr);
  BIND_DIVIDE_PTR(SSparseArrayDouble2dPtr);

  DEF_FN(test_VArrayDouble_append1, VArrayDoublePtr (*)(int));
  DEF_FN(test_VArrayDouble_append,
         VArrayDoublePtr (*)(VArrayDoublePtr, SArrayDoublePtr));
  DEF_FN(test_sort_inplace_ArrayDouble, void (*)(ArrayDouble &, bool));
  DEF_FN(test_sort_ArrayDouble, SArrayDoublePtr (*)(ArrayDouble &, bool));
  DEF_FN(test_sort_index_inplace_ArrayDouble,
         void (*)(ArrayDouble &, ArrayULong &, bool));
  DEF_FN(test_sort_index_ArrayDouble,
         SArrayDoublePtr (*)(ArrayDouble &, ArrayULong &, bool));
  DEF_FN(test_sort_abs_index_inplace_ArrayDouble,
         void (*)(ArrayDouble &, ArrayULong &, bool));
  DEF_FN(test_mult_incr_ArrayDouble,
         void (*)(ArrayDouble &, BaseArrayDouble &, double));
  DEF_FN(test_mult_fill_ArrayDouble,
         void (*)(ArrayDouble &, BaseArrayDouble &, double));
  DEF_FN(test_mult_add_mult_incr_ArrayDouble,
         void (*)(ArrayDouble &, BaseArrayDouble &, double, BaseArrayDouble &,
                  double));
}

void bind_performance_tests(py::module_ &m) {
  DEF_FN(test_sum_double_pointer, double (*)(ulong, ulong));
  DEF_FN(test_sum_ArrayDouble, double (*)(ulong, ulong));
  DEF_FN(test_sum_SArray_shared_ptr, double (*)(ulong, ulong));
  DEF_FN(test_sum_VArray_shared_ptr, double (*)(ulong, ulong));
}

#define BIND_TYPEMAP_FUNCTIONS(                                                \
    TYPE, SUFFIX, ARRAY_TYPE, ARRAY2D_TYPE, ARRAYLIST1D_TYPE,                  \
    ARRAYLIST2D_TYPE, SPARSEARRAY_TYPE, SPARSEARRAY2D_TYPE,                    \
    SARRAYPTR_TYPE, SARRAY2DPTR_TYPE, SARRAYPTRLIST1D_TYPE,                    \
    SARRAYPTRLIST2D_TYPE, SARRAY2DPTR_LIST1D_TYPE,                             \
    SARRAY2DPTR_LIST2D_TYPE, VARRAYPTR_TYPE, VARRAYPTRLIST1D_TYPE,             \
    VARRAYPTRLIST2D_TYPE, BASEARRAY_TYPE, BASEARRAY2D_TYPE,                    \
    SSPARSEARRAYPTR_TYPE, SSPARSEARRAY2DPTR_TYPE, SBASEARRAYPTR_TYPE,          \
    SBASEARRAY2DPTR_TYPE, BASEARRAY_LIST1D_TYPE, BASEARRAY_LIST2D_TYPE,        \
    BASEARRAY2D_LIST1D_TYPE, BASEARRAY2D_LIST2D_TYPE,                          \
    SBASEARRAYPTR_LIST1D_TYPE, SBASEARRAYPTR_LIST2D_TYPE,                      \
    SBASEARRAY2DPTR_LIST1D_TYPE, SBASEARRAY2DPTR_LIST2D_TYPE)                  \
  DEF_FN(test_typemap_in_##ARRAY_TYPE, TYPE (*)(ARRAY_TYPE &));                \
  DEF_FN(test_typemap_in_##ARRAY_TYPE, TYPE (*)(TYPE));                        \
  DEF_FN(test_typemap_in_##ARRAY2D_TYPE, TYPE (*)(ARRAY2D_TYPE &));            \
  DEF_FN(test_typemap_in_##ARRAY2D_TYPE, TYPE (*)(TYPE));                      \
  DEF_FN(test_typemap_in_##ARRAYLIST1D_TYPE, TYPE (*)(ARRAYLIST1D_TYPE &));    \
  DEF_FN(test_typemap_in_##ARRAYLIST1D_TYPE, TYPE (*)(TYPE));                  \
  DEF_FN(test_typemap_in_##ARRAYLIST2D_TYPE, TYPE (*)(ARRAYLIST2D_TYPE &));    \
  DEF_FN(test_typemap_in_##ARRAYLIST2D_TYPE, TYPE (*)(TYPE));                  \
  DEF_FN(test_typemap_in_##SPARSEARRAY_TYPE, TYPE (*)(SPARSEARRAY_TYPE &));    \
  DEF_FN(test_typemap_in_##SPARSEARRAY_TYPE, TYPE (*)(TYPE));                  \
  DEF_FN(test_typemap_in_##SPARSEARRAY2D_TYPE, TYPE (*)(SPARSEARRAY2D_TYPE &));\
  DEF_FN(test_typemap_in_##SPARSEARRAY2D_TYPE, TYPE (*)(TYPE));                \
  DEF_FN(test_typemap_in_##SARRAYPTR_TYPE, TYPE (*)(SARRAYPTR_TYPE));          \
  DEF_FN(test_typemap_in_##SARRAYPTR_TYPE, TYPE (*)(TYPE));                    \
  DEF_FN(test_typemap_in_##SARRAY2DPTR_TYPE, TYPE (*)(SARRAY2DPTR_TYPE));      \
  DEF_FN(test_typemap_in_##SARRAY2DPTR_TYPE, TYPE (*)(TYPE));                  \
  DEF_FN(test_typemap_in_##SARRAYPTRLIST1D_TYPE,                               \
         TYPE (*)(SARRAYPTRLIST1D_TYPE &));                                    \
  DEF_FN(test_typemap_in_##SARRAYPTRLIST1D_TYPE, TYPE (*)(TYPE));              \
  DEF_FN(test_typemap_in_##SARRAYPTRLIST2D_TYPE,                               \
         TYPE (*)(SARRAYPTRLIST2D_TYPE &));                                    \
  DEF_FN(test_typemap_in_##SARRAYPTRLIST2D_TYPE, TYPE (*)(TYPE));              \
  DEF_FN(test_typemap_in_##SARRAY2DPTR_LIST1D_TYPE,                            \
         TYPE (*)(SARRAY2DPTR_LIST1D_TYPE &));                                 \
  DEF_FN(test_typemap_in_##SARRAY2DPTR_LIST1D_TYPE, TYPE (*)(TYPE));           \
  DEF_FN(test_typemap_in_##SARRAY2DPTR_LIST2D_TYPE,                            \
         TYPE (*)(SARRAY2DPTR_LIST2D_TYPE &));                                 \
  DEF_FN(test_typemap_in_##SARRAY2DPTR_LIST2D_TYPE, TYPE (*)(TYPE));           \
  DEF_FN(test_typemap_in_##VARRAYPTR_TYPE, TYPE (*)(VARRAYPTR_TYPE));          \
  DEF_FN(test_typemap_in_##VARRAYPTR_TYPE, TYPE (*)(TYPE));                    \
  DEF_FN(test_typemap_in_##VARRAYPTRLIST1D_TYPE,                               \
         TYPE (*)(VARRAYPTRLIST1D_TYPE &));                                    \
  DEF_FN(test_typemap_in_##VARRAYPTRLIST1D_TYPE, TYPE (*)(TYPE));              \
  DEF_FN(test_typemap_in_##VARRAYPTRLIST2D_TYPE,                               \
         TYPE (*)(VARRAYPTRLIST2D_TYPE &));                                    \
  DEF_FN(test_typemap_in_##VARRAYPTRLIST2D_TYPE, TYPE (*)(TYPE));              \
  DEF_FN(test_typemap_in_##BASEARRAY_TYPE, TYPE (*)(BASEARRAY_TYPE &));        \
  DEF_FN(test_typemap_in_##BASEARRAY_TYPE, TYPE (*)(TYPE));                    \
  DEF_FN(test_typemap_in_##BASEARRAY2D_TYPE, TYPE (*)(BASEARRAY2D_TYPE &));    \
  DEF_FN(test_typemap_in_##BASEARRAY2D_TYPE, TYPE (*)(TYPE));                  \
  DEF_FN(test_typemap_in_##SSPARSEARRAYPTR_TYPE,                               \
         TYPE (*)(SSPARSEARRAYPTR_TYPE));                                      \
  DEF_FN(test_typemap_in_##SSPARSEARRAYPTR_TYPE, TYPE (*)(TYPE));              \
  DEF_FN(test_typemap_in_##SSPARSEARRAY2DPTR_TYPE,                             \
         TYPE (*)(SSPARSEARRAY2DPTR_TYPE));                                    \
  DEF_FN(test_typemap_in_##SSPARSEARRAY2DPTR_TYPE, TYPE (*)(TYPE));            \
  DEF_FN(test_typemap_in_##SBASEARRAYPTR_TYPE, TYPE (*)(SBASEARRAYPTR_TYPE));  \
  DEF_FN(test_typemap_in_##SBASEARRAYPTR_TYPE, TYPE (*)(TYPE));                \
  DEF_FN(test_typemap_in_##SBASEARRAY2DPTR_TYPE,                               \
         TYPE (*)(SBASEARRAY2DPTR_TYPE));                                      \
  DEF_FN(test_typemap_in_##SBASEARRAY2DPTR_TYPE, TYPE (*)(TYPE));              \
  DEF_FN(test_typemap_in_##BASEARRAY_LIST1D_TYPE,                              \
         TYPE (*)(BASEARRAY_LIST1D_TYPE &));                                   \
  DEF_FN(test_typemap_in_##BASEARRAY_LIST1D_TYPE, TYPE (*)(TYPE));             \
  DEF_FN(test_typemap_in_##BASEARRAY_LIST2D_TYPE,                              \
         TYPE (*)(BASEARRAY_LIST2D_TYPE &));                                   \
  DEF_FN(test_typemap_in_##BASEARRAY_LIST2D_TYPE, TYPE (*)(TYPE));             \
  DEF_FN(test_typemap_in_##BASEARRAY2D_LIST1D_TYPE,                            \
         TYPE (*)(BASEARRAY2D_LIST1D_TYPE &));                                 \
  DEF_FN(test_typemap_in_##BASEARRAY2D_LIST1D_TYPE, TYPE (*)(TYPE));           \
  DEF_FN(test_typemap_in_##BASEARRAY2D_LIST2D_TYPE,                            \
         TYPE (*)(BASEARRAY2D_LIST2D_TYPE &));                                 \
  DEF_FN(test_typemap_in_##BASEARRAY2D_LIST2D_TYPE, TYPE (*)(TYPE));           \
  DEF_FN(test_typemap_in_##SBASEARRAYPTR_LIST1D_TYPE,                          \
         TYPE (*)(SBASEARRAYPTR_LIST1D_TYPE &));                               \
  DEF_FN(test_typemap_in_##SBASEARRAYPTR_LIST1D_TYPE, TYPE (*)(TYPE));         \
  DEF_FN(test_typemap_in_##SBASEARRAYPTR_LIST2D_TYPE,                          \
         TYPE (*)(SBASEARRAYPTR_LIST2D_TYPE &));                               \
  DEF_FN(test_typemap_in_##SBASEARRAYPTR_LIST2D_TYPE, TYPE (*)(TYPE));         \
  DEF_FN(test_typemap_in_##SBASEARRAY2DPTR_LIST1D_TYPE,                        \
         TYPE (*)(SBASEARRAY2DPTR_LIST1D_TYPE &));                             \
  DEF_FN(test_typemap_in_##SBASEARRAY2DPTR_LIST1D_TYPE, TYPE (*)(TYPE));       \
  DEF_FN(test_typemap_in_##SBASEARRAY2DPTR_LIST2D_TYPE,                        \
         TYPE (*)(SBASEARRAY2DPTR_LIST2D_TYPE &));                             \
  DEF_FN(test_typemap_in_##SBASEARRAY2DPTR_LIST2D_TYPE, TYPE (*)(TYPE));       \
  DEF_FN(test_typemap_out_##SARRAYPTR_TYPE, SARRAYPTR_TYPE (*)(ulong));        \
  DEF_FN(test_typemap_out_##SARRAYPTRLIST1D_TYPE,                              \
         SARRAYPTRLIST1D_TYPE (*)(int));                                       \
  DEF_FN(test_typemap_out_##SARRAYPTRLIST2D_TYPE,                              \
         SARRAYPTRLIST2D_TYPE (*)(int, int));                                  \
  DEF_FN(test_typemap_out_##SARRAY2DPTR_TYPE,                                  \
         SARRAY2DPTR_TYPE (*)(ulong, ulong));                                  \
  bind_typemap_validators<TYPE>(m, #SUFFIX)

void bind_typemap_tests(py::module_ &m) {
  BIND_TYPEMAP_FUNCTIONS(
      double, Double, ArrayDouble, ArrayDouble2d, ArrayDoubleList1D,
      ArrayDoubleList2D, SparseArrayDouble, SparseArrayDouble2d,
      SArrayDoublePtr, SArrayDouble2dPtr, SArrayDoublePtrList1D,
      SArrayDoublePtrList2D, SArrayDouble2dPtrList1D,
      SArrayDouble2dPtrList2D, VArrayDoublePtr, VArrayDoublePtrList1D,
      VArrayDoublePtrList2D, BaseArrayDouble, BaseArrayDouble2d,
      SSparseArrayDoublePtr, SSparseArrayDouble2dPtr, SBaseArrayDoublePtr,
      SBaseArrayDouble2dPtr, BaseArrayDoubleList1D, BaseArrayDoubleList2D,
      BaseArrayDouble2dList1D, BaseArrayDouble2dList2D,
      SBaseArrayDoublePtrList1D, SBaseArrayDoublePtrList2D,
      SBaseArrayDouble2dPtrList1D, SBaseArrayDouble2dPtrList2D);

  BIND_TYPEMAP_FUNCTIONS(
      std::int32_t, Int, ArrayInt, ArrayInt2d, ArrayIntList1D, ArrayIntList2D,
      SparseArrayInt, SparseArrayInt2d, SArrayIntPtr, SArrayInt2dPtr,
      SArrayIntPtrList1D, SArrayIntPtrList2D, SArrayInt2dPtrList1D,
      SArrayInt2dPtrList2D, VArrayIntPtr, VArrayIntPtrList1D,
      VArrayIntPtrList2D, BaseArrayInt, BaseArrayInt2d, SSparseArrayIntPtr,
      SSparseArrayInt2dPtr, SBaseArrayIntPtr, SBaseArrayInt2dPtr,
      BaseArrayIntList1D, BaseArrayIntList2D, BaseArrayInt2dList1D,
      BaseArrayInt2dList2D, SBaseArrayIntPtrList1D, SBaseArrayIntPtrList2D,
      SBaseArrayInt2dPtrList1D, SBaseArrayInt2dPtrList2D);

  BIND_TYPEMAP_FUNCTIONS(
      std::int16_t, Short, ArrayShort, ArrayShort2d, ArrayShortList1D,
      ArrayShortList2D, SparseArrayShort, SparseArrayShort2d, SArrayShortPtr,
      SArrayShort2dPtr, SArrayShortPtrList1D, SArrayShortPtrList2D,
      SArrayShort2dPtrList1D, SArrayShort2dPtrList2D, VArrayShortPtr,
      VArrayShortPtrList1D, VArrayShortPtrList2D, BaseArrayShort,
      BaseArrayShort2d, SSparseArrayShortPtr, SSparseArrayShort2dPtr,
      SBaseArrayShortPtr, SBaseArrayShort2dPtr, BaseArrayShortList1D,
      BaseArrayShortList2D, BaseArrayShort2dList1D, BaseArrayShort2dList2D,
      SBaseArrayShortPtrList1D, SBaseArrayShortPtrList2D,
      SBaseArrayShort2dPtrList1D, SBaseArrayShort2dPtrList2D);

  BIND_TYPEMAP_FUNCTIONS(
      std::uint16_t, UShort, ArrayUShort, ArrayUShort2d, ArrayUShortList1D,
      ArrayUShortList2D, SparseArrayUShort, SparseArrayUShort2d,
      SArrayUShortPtr, SArrayUShort2dPtr, SArrayUShortPtrList1D,
      SArrayUShortPtrList2D, SArrayUShort2dPtrList1D,
      SArrayUShort2dPtrList2D, VArrayUShortPtr, VArrayUShortPtrList1D,
      VArrayUShortPtrList2D, BaseArrayUShort, BaseArrayUShort2d,
      SSparseArrayUShortPtr, SSparseArrayUShort2dPtr, SBaseArrayUShortPtr,
      SBaseArrayUShort2dPtr, BaseArrayUShortList1D, BaseArrayUShortList2D,
      BaseArrayUShort2dList1D, BaseArrayUShort2dList2D,
      SBaseArrayUShortPtrList1D, SBaseArrayUShortPtrList2D,
      SBaseArrayUShort2dPtrList1D, SBaseArrayUShort2dPtrList2D);

  BIND_TYPEMAP_FUNCTIONS(
      std::int64_t, Long, ArrayLong, ArrayLong2d, ArrayLongList1D,
      ArrayLongList2D, SparseArrayLong, SparseArrayLong2d, SArrayLongPtr,
      SArrayLong2dPtr, SArrayLongPtrList1D, SArrayLongPtrList2D,
      SArrayLong2dPtrList1D, SArrayLong2dPtrList2D, VArrayLongPtr,
      VArrayLongPtrList1D, VArrayLongPtrList2D, BaseArrayLong,
      BaseArrayLong2d, SSparseArrayLongPtr, SSparseArrayLong2dPtr,
      SBaseArrayLongPtr, SBaseArrayLong2dPtr, BaseArrayLongList1D,
      BaseArrayLongList2D, BaseArrayLong2dList1D, BaseArrayLong2dList2D,
      SBaseArrayLongPtrList1D, SBaseArrayLongPtrList2D,
      SBaseArrayLong2dPtrList1D, SBaseArrayLong2dPtrList2D);

  BIND_TYPEMAP_FUNCTIONS(
      std::uint64_t, ULong, ArrayULong, ArrayULong2d, ArrayULongList1D,
      ArrayULongList2D, SparseArrayULong, SparseArrayULong2d, SArrayULongPtr,
      SArrayULong2dPtr, SArrayULongPtrList1D, SArrayULongPtrList2D,
      SArrayULong2dPtrList1D, SArrayULong2dPtrList2D, VArrayULongPtr,
      VArrayULongPtrList1D, VArrayULongPtrList2D, BaseArrayULong,
      BaseArrayULong2d, SSparseArrayULongPtr, SSparseArrayULong2dPtr,
      SBaseArrayULongPtr, SBaseArrayULong2dPtr, BaseArrayULongList1D,
      BaseArrayULongList2D, BaseArrayULong2dList1D, BaseArrayULong2dList2D,
      SBaseArrayULongPtrList1D, SBaseArrayULongPtrList2D,
      SBaseArrayULong2dPtrList1D, SBaseArrayULong2dPtrList2D);

  BIND_TYPEMAP_FUNCTIONS(
      std::uint32_t, UInt, ArrayUInt, ArrayUInt2d, ArrayUIntList1D,
      ArrayUIntList2D, SparseArrayUInt, SparseArrayUInt2d, SArrayUIntPtr,
      SArrayUInt2dPtr, SArrayUIntPtrList1D, SArrayUIntPtrList2D,
      SArrayUInt2dPtrList1D, SArrayUInt2dPtrList2D, VArrayUIntPtr,
      VArrayUIntPtrList1D, VArrayUIntPtrList2D, BaseArrayUInt,
      BaseArrayUInt2d, SSparseArrayUIntPtr, SSparseArrayUInt2dPtr,
      SBaseArrayUIntPtr, SBaseArrayUInt2dPtr, BaseArrayUIntList1D,
      BaseArrayUIntList2D, BaseArrayUInt2dList1D, BaseArrayUInt2dList2D,
      SBaseArrayUIntPtrList1D, SBaseArrayUIntPtrList2D,
      SBaseArrayUInt2dPtrList1D, SBaseArrayUInt2dPtrList2D);

  BIND_TYPEMAP_FUNCTIONS(
      float, Float, ArrayFloat, ArrayFloat2d, ArrayFloatList1D,
      ArrayFloatList2D, SparseArrayFloat, SparseArrayFloat2d, SArrayFloatPtr,
      SArrayFloat2dPtr, SArrayFloatPtrList1D, SArrayFloatPtrList2D,
      SArrayFloat2dPtrList1D, SArrayFloat2dPtrList2D, VArrayFloatPtr,
      VArrayFloatPtrList1D, VArrayFloatPtrList2D, BaseArrayFloat,
      BaseArrayFloat2d, SSparseArrayFloatPtr, SSparseArrayFloat2dPtr,
      SBaseArrayFloatPtr, SBaseArrayFloat2dPtr, BaseArrayFloatList1D,
      BaseArrayFloatList2D, BaseArrayFloat2dList1D, BaseArrayFloat2dList2D,
      SBaseArrayFloatPtrList1D, SBaseArrayFloatPtrList2D,
      SBaseArrayFloat2dPtrList1D, SBaseArrayFloat2dPtrList2D);
}

void bind_container_helpers(py::module_ &m) {
  py::class_<VarrayContainer>(m, "VarrayContainer")
      .def(py::init<>())
      .def("initVarray", static_cast<void (VarrayContainer::*)()>(
                              &VarrayContainer::initVarray))
      .def("initVarray", static_cast<void (VarrayContainer::*)(int)>(
                              &VarrayContainer::initVarray),
           py::arg("size"))
      .def("nRef", &VarrayContainer::nRef)
      .def_readwrite("varrayPtr", &VarrayContainer::varrayPtr);

  py::class_<VarrayUser>(m, "VarrayUser")
      .def(py::init<>())
      .def("nRef", &VarrayUser::nRef)
      .def("setArray", &VarrayUser::setArray, py::arg("vcc"));

  DEF_FN(test_sbasearray_container_new, void (*)(SBaseArrayDoublePtr));
  DEF_FN(test_sbasearray_container_clear, void (*)());
  DEF_FN(test_sbasearray_container_compute, double (*)());
  DEF_FN(test_sbasearray2d_container_new, void (*)(SBaseArrayDouble2dPtr));
  DEF_FN(test_sbasearray2d_container_clear, void (*)());
  DEF_FN(test_sbasearray2d_container_compute, double (*)());
}

}  // namespace

PYBIND11_MODULE(array_test, m) {
  tick::pybind::ensure_numpy_imported();

  m.doc() = "tick.array_test pybind11 bindings";

  bind_container_helpers(m);
  bind_array_method_tests(m);
  bind_performance_tests(m);
  bind_typemap_tests(m);
}
