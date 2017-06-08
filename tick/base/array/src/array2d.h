#ifndef TICK_BASE_ARRAY_SRC_ARRAY2D_H_
#define TICK_BASE_ARRAY_SRC_ARRAY2D_H_

/** @file */

#include "defs.h"
#include "alloc.h"
#include "basearray2d.h"

template<typename T>
class SArray2d;
template<typename T>
class SparseArray2d;

/*! \class Array2d
 * \brief Template class for basic non sparse 2d arrays of type `T`.
 *
 * It manages the fact that allocation can be owned by a different object. At this level, if
 * it is not self-owned, the owner could ony be another C structure.
 * It is important to understand that if you need the array to be shared with the Python interpreter
 * then this is ONLY handled by SArray classes through their smart pointers `SArrayPtr`.
 *
 * In order to use this class, you have to understand clearly when copy of the data is or is not
 * performed. Here is a small example.
 *
 *      ArrayDouble2d c(10, 10); // Creates an array of double of size 10x10
 *      ArrayDouble b = c; // Copies the data
 *      ArrayDouble e = view(c) // Does not copy the data
 *      b = view(c) // No copy
 *
 */
template<typename T>
class Array2d : public BaseArray2d<T> {
 protected:
    using BaseArray2d<T>::_size;
    using BaseArray2d<T>::is_data_allocation_owned;
    using BaseArray2d<T>::is_indices_allocation_owned;
    using BaseArray2d<T>::_data;
    using BaseArray2d<T>::_indices;
    using BaseArray2d<T>::_row_indices;
    using BaseArray2d<T>::_n_cols;
    using BaseArray2d<T>::_n_rows;

    // NB: We always have _n_rows * _n_cols == _size

 public:
    using AbstractArray1d2d<T>::is_dense;
    using AbstractArray1d2d<T>::is_sparse;
    using AbstractArray1d2d<T>::init_to_zero;

    //! @brief Constructor for an empty array.
    Array2d() : BaseArray2d<T>(true) {}

    /**
     * @brief Constructor for constructing a 2d array of size `n_rows, n_cols`
     * (and eventually some preallocated data).
     *
     * \param n_rows The number of rows
     * \param n_cols The number of cols
     * \param data A pointer to an array of elements of type `T`. Default is
     * `nullptr`. This pointer should point to allocation created by macro
     * PYSHARED_ALLOC_ARRAY(and desallocated with PYSHARED_FREE_ARRAY)
     *
     * \return If `data == nullptr`, then a zero-valued array is created.
     * Otherwise `data` will be used for the data of the array.
     * \warning If data != nullptr then the created object does not own the allocation
     */
    explicit Array2d(ulong n_rows, ulong n_cols, T *data = nullptr);

    //! @brief The copy constructor
    Array2d(const Array2d<T> &other) = default;

    //! @brief The move constructor
    Array2d(Array2d<T> &&other) = default;

    //! @brief The copy assignement operator
    Array2d<T> &operator=(const Array2d<T> &other) = default;

    //! @brief The move assignement operator
    Array2d<T> &operator=(Array2d<T> &&other) = default;

    //! @brief Destructor
    virtual ~Array2d() {}

    //! @brief Fill vector with given value
    void fill(T value);

    //! @brief Multiply matrix x by factor a inplace and increment this by new matrix
    //! @brief Namely, we perform the operator this += x * a
    //! \param x : an Array2d
    //! \param a : a scalar of type T
    //! @note Scalar is of type T, meaning that real values will get truncated before
    //! multiplication if T is an integer type
    void mult_incr(const Array2d<T>& x, const T a);

    //! @brief Multiply matrix x by factor a inplace and fill using this by new matrix
    //! @brief Namely, we perform the operator this = x * a
    //! \param x : an Array2d
    //! \param a : a scalar
    //! @note Scalar is of type T, meaning that real values will get truncated before
    //! multiplication if T is an integer type
    void mult_fill(const Array2d<T>& x, const T a);

    //! @brief Multiply matrix x by factor c inplace and fill using this by new matrix
    //! @brief Namely, we perform the operator this += a * x + b * y
    //! \param x : an Array2d
    //! \param a : a scalar
    //! \param y : an Array2d
    //! \param b : a scalar
    void mult_add_mult_incr(const Array2d<T>& x, const T a, const Array2d<T>& y, const T b);

    /**
     * @brief Bracket notation for extraction and assignements. We use 1d index.
     *
     * \param i Index in the array (i = r*n_cols+c, where r (c) is the row (column) number
     * \return A reference to the element number \a i
     */
    inline T &operator[](const ulong i) {
#ifdef DEBUG_COSTLY_THROW
        if (i >= _size) TICK_BAD_INDEX(0, _size, i);
#endif
        return _data[i];
    }

    //! @brief Get a reference to a value
    //! \param row The row number
    //! \param col The column number
    //! \return A reference to the corresponding
    inline T &value(ulong row, ulong col) {
#ifdef DEBUG_COSTLY_THROW
        if (row >= _n_rows) TICK_BAD_INDEX(0, row, _n_rows);
        if (col >= _n_cols) TICK_BAD_INDEX(0, col, _n_cols);
#endif
        return _data[row * _n_cols + col];
    }

    //! @brief Returns a shared pointer to a SArray2d encapsulating the array
    //! \warning : The ownership of the data is given to the returned structure
    //! THUS the array *this becomes a view.
    //! \warning : This method cannot be called on a view
    // The definition is in the file sarray.h
    std::shared_ptr<SArray2d<T>> as_sarray2d_ptr();
};

// Constructor
template<typename T>
Array2d<T>::Array2d(ulong n_rows, ulong n_cols, T *data) :
    BaseArray2d<T>(true) {
#ifdef DEBUG_ARRAY
    std::cout << "Array2d Constructor : Array(n_rows=" << n_rows << ", n_cols=" << n_cols << ",data=" << data << ") --> " << this << std::endl;
#endif
    _n_cols = n_cols;
    _n_rows = n_rows;
    _size = n_cols * n_rows;
    // if no one gave us data we allocate it and are now responsible for it
    if (data == nullptr) {
        is_data_allocation_owned = true;
        TICK_PYTHON_MALLOC(_data, T, _size);
    } else {
        // Otherwise the one who gave the data is responsible for its allocation
        is_data_allocation_owned = false;
        _data = data;
    }
}

// fill with given value
template<typename T>
void Array2d<T>::fill(T value) {
    tick::vector_operations<T>{}.set(_size, value, _data);
}

template<typename T>
void Array2d<T>::mult_incr(const Array2d<T>& x, const T a) {
    Array<T> this_array = Array<T>(this->size(), this->data());
    Array<T> x_array = Array<T>(x.size(), x.data());
    this_array.mult_incr(x_array, a);
}

template<typename T>
void Array2d<T>::mult_fill(const Array2d<T>& x, const T a) {
    Array<T> this_array = Array<T>(this->size(), this->data());
    Array<T> x_array = Array<T>(x.size(), x.data());
    this_array.mult_fill(x_array, a);
}

template<typename T>
void Array2d<T>::mult_add_mult_incr(const Array2d<T>& x, const T a, const Array2d<T>& y,
                                    const T b) {
    Array<T> this_array = Array<T>(this->size(), this->data());
    Array<T> x_array = Array<T>(x.size(), x.data());
    Array<T> y_array = Array<T>(y.size(), y.data());
    this_array.mult_add_mult_incr(x_array, a, y_array, b);
}

/**
 * Array2d serialization function for binary archives types
 */
template <class Archive, class T>
typename std::enable_if<cereal::traits::is_output_serializable<cereal::BinaryData<T>, Archive>::value, void>::type
CEREAL_SAVE_FUNCTION_NAME(Archive & ar, BaseArray2d<T> const & arr) {
    const bool is_sparse = arr.is_sparse();

    ar(CEREAL_NVP(is_sparse));

    ar(CEREAL_NVP(arr.n_cols()));
    ar(CEREAL_NVP(arr.n_rows()));

    ar(cereal::make_size_tag(arr.size()));
    ar(cereal::binary_data(arr.data(), arr.size() * sizeof(T)));

    if (is_sparse) {
        ar(cereal::make_size_tag(arr.size()));
        ar(cereal::binary_data(arr.indices(), arr.size() * sizeof(ulong)));
    }
}

/**
 * Array2d serialization function for text archives types (XML, JSON)
 */
template <class Archive, class T>
typename std::enable_if<!cereal::traits::is_output_serializable<cereal::BinaryData<T>, Archive>::value, void>::type
CEREAL_SAVE_FUNCTION_NAME(Archive & ar, BaseArray2d<T> const & arr) {
    const bool is_sparse = arr.is_sparse();
    const ulong n_cols = arr.n_cols();
    const ulong n_rows = arr.n_rows();

    ar(CEREAL_NVP(is_sparse));

    ar(CEREAL_NVP(n_cols));
    ar(CEREAL_NVP(n_rows));

    {
        ar.setNextName("values");
        ar.startNode();

        ar(cereal::make_size_tag(arr.size_data()));

        for (ulong i = 0; i < arr.size_data(); ++i)
            ar(arr.data()[i]);

        ar.finishNode();
    }

    if (is_sparse) {
        ar.setNextName("indices");
        ar.startNode();

        ar(cereal::make_size_tag(arr.size_sparse()));

        for (ulong i = 0; i < arr.size_sparse(); ++i)
            ar(arr.indices()[i]);

        ar.finishNode();
    }
}

/**
 * Array2d deserialization function for binary archives types
 */
template <class Archive, class T>
typename std::enable_if<cereal::traits::is_input_serializable<cereal::BinaryData<T>, Archive>::value, void>::type
CEREAL_LOAD_FUNCTION_NAME(Archive & ar, BaseArray2d<T> & arr) {
    bool is_sparse = false;
    ulong n_cols = 0;
    ulong n_rows = 0;

    ar(is_sparse);
    ar(n_cols);
    ar(n_rows);

    ulong vectorSize = 0;
    ar(cereal::make_size_tag(vectorSize));

    if (vectorSize != n_cols * n_rows)
      TICK_ERROR("Bad format in array 2d deserrialization (size="
                     << vectorSize << ", n_rows=" << n_rows << ", n_cols=" << n_cols << ")");

    arr = Array2d<T>(n_rows, n_cols);
    ar(cereal::binary_data(arr.data(), static_cast<std::size_t>( vectorSize ) * sizeof(T)));

    if (is_sparse)
        TICK_ERROR("Deserializing sparse arrays is not supported yet.");
}

/**
* Array2d deserialization function for text archives types (XML, JSON)
*/
template <class Archive, class T>
typename std::enable_if<!cereal::traits::is_input_serializable<cereal::BinaryData<T>, Archive>::value, void>::type
CEREAL_LOAD_FUNCTION_NAME(Archive & ar, BaseArray2d<T> & arr) {
    bool is_sparse = false;
    ulong n_cols = 0;
    ulong n_rows = 0;

    ar(CEREAL_NVP(is_sparse));
    ar(CEREAL_NVP(n_cols));
    ar(CEREAL_NVP(n_rows));

    {
        ar.setNextName("values");
        ar.startNode();

        ulong vectorSize;
        ar(cereal::make_size_tag(vectorSize));

        arr = Array2d<T>(n_rows, n_cols);

        for (ulong i = 0; i < arr.size_data(); ++i)
            ar(arr.data()[i]);

        ar.finishNode();
    }

    if (is_sparse)
        TICK_ERROR("Deserializing sparse arrays is not supported yet.");
}

/////////////////////////////////////////////////////////////////
//
//  The various instances of this template
//
/////////////////////////////////////////////////////////////////

/** @defgroup array2d_sub_mod The instantiations of the Array2d template
 *  @ingroup Array_typedefs_mod
 * @{
 */

typedef Array2d<double> ArrayDouble2d;
typedef Array2d<float> ArrayFloat2d;
typedef Array2d<std::int32_t> ArrayInt2d;
typedef Array2d<std::uint32_t> ArrayUInt2d;
typedef Array2d<std::int16_t> ArrayShort2d;
typedef Array2d<std::uint16_t> ArrayUShort2d;
typedef Array2d<std::int64_t> ArrayLong2d;
typedef Array2d<ulong> ArrayULong2d;

/**
 * @}
 */

/** @defgroup array2dlist1d_sub_mod The classes for dealing with 1d-list of Arrays2d
 *  @ingroup Array_typedefs_mod
 * @{
 */

typedef std::vector<Array2d<float> > ArrayFloat2dList1D;
typedef std::vector<Array2d<double> > ArrayDouble2dList1D;
typedef std::vector<Array2d<std::int32_t> > ArrayInt2dList1D;
typedef std::vector<Array2d<std::uint32_t> > ArrayUInt2dList1D;
typedef std::vector<Array2d<std::int16_t> > ArrayShort2dList1D;
typedef std::vector<Array2d<std::uint16_t> > ArrayUShort2dList1D;
typedef std::vector<Array2d<std::int64_t> > ArrayLong2dList1D;
typedef std::vector<Array2d<ulong> > ArrayULong2dList1D;


/**
 * @}
 */


/** @defgroup array2dlist2d_sub_mod The classes for dealing with 2d-list of Arrays2d
 *  @ingroup Array_typedefs_mod
 * @{
 */

typedef std::vector<std::vector<Array2d<float> > > ArrayFloat2dList2D;
typedef std::vector<std::vector<Array2d<double> > > ArrayDouble2dList2D;
typedef std::vector<std::vector<Array2d<std::int32_t> > > ArrayInt2dList2D;
typedef std::vector<std::vector<Array2d<std::uint32_t> > > ArrayUInt2dList2D;
typedef std::vector<std::vector<Array2d<std::int16_t> > > ArrayShort2dList2D;
typedef std::vector<std::vector<Array2d<std::uint16_t> > > ArrayUShort2dList2D;
typedef std::vector<std::vector<Array2d<std::int64_t> > > ArrayLong2dList2D;
typedef std::vector<std::vector<Array2d<ulong> > > ArrayULong2dList2D;


/**
 * @}
 */


/**
 * Output function to log.
 *
 * Usage example:
 *
 * ArrayDouble2d d(10, 10);
 * TICK_DEBUG() << "MyArray: " << d;
 */
template <typename E, typename T>
tick::TemporaryLog<E>& operator<<(tick::TemporaryLog<E>& log, const Array2d<T>& arr) {
    const auto size = arr.size();
    const auto n_cols = arr.n_cols();
    const auto n_rows = arr.n_rows();

    log << "Array2d[nrows=" << arr.n_rows() << ", ncols=" << arr.n_cols() << ", " << typeid(T).name();

    log << "\n[";

    const ulong max_rows = 8;
    const ulong max_cols = max_rows;

    for (auto r = 0; r < std::min(max_rows, n_rows); ++r) {
        log << "[";

        for (auto c = 0; c < std::min(max_cols, n_cols); ++c) {
            log << arr.data()[r * n_cols + c] << ", ";
        }

        if (n_cols > max_cols) {
            log << "... ";
        }

        log << "]\n";
    }

    if (n_rows > max_rows) {
        log << "... ]]";
    } else {
        log << "]]";
    }

    return log;
}

#endif  // TICK_BASE_ARRAY_SRC_ARRAY2D_H_
