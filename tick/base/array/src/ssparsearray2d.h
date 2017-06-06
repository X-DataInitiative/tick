#ifndef TICK_BASE_ARRAY_SRC_SSPARSEARRAY2D_H_
#define TICK_BASE_ARRAY_SRC_SSPARSEARRAY2D_H_

//
//  ssparsearray2d.h
//  TICK
//

/** @file */

#include <memory>
#include "sparsearray2d.h"

///////////////////////////////////////////////////////////////////////////////////
//
//  SSparseArray2d
//
///////////////////////////////////////////////////////////////////////////////////

/*! \class SSparseArray2d
 * \brief Template class for 2d sparse arrays of type `T` fully shareable
 * with Python.
 * This is the array class that introduces the ability to share its allocation with Python.
 * \warning You should always use the associated smart pointer class
 * `SSparseArray2dPtr` and never access this class directly. Thus the only constructor
 * to be used is the `SSparseArray2d<T>::new_ptr()` constructor. In that way you can also share the
 * allocations within C++ (using a reference counter).
 */

template<typename T>
class SSparseArray2d : public SparseArray2d<T> {
 protected:
    using SparseArray2d<T>::_size;
    using SparseArray2d<T>::_n_rows;
    using SparseArray2d<T>::_n_cols;
    using SparseArray2d<T>::is_data_allocation_owned;
    using SparseArray2d<T>::_data;
    using SparseArray2d<T>::_size_sparse;
    using SparseArray2d<T>::is_indices_allocation_owned;
    using SparseArray2d<T>::_indices;
    using SparseArray2d<T>::is_row_indices_allocation_owned;
    using SparseArray2d<T>::_row_indices;

#ifdef PYTHON_LINK
    //! @brief The (eventual) Python owner of the array _data;
    //! If ==nullptr then it is self-owned
    void *_data_owner;

    //! @brief The (eventual) Python owner of the array _indices;
    //! If ==nullptr then it is self-owned
    void *_indices_owner;

    //! @brief The (eventual) Python owner of the array _row_indices;
    //! If ==nullptr then it is self-owned
    void *_row_indices_owner;
#endif

 public:
#ifdef PYTHON_LINK
    void give_data_indices_rowindices_owners(void *data_owner, void* indices_owner, void * rowindices_owner);

    //! @brief returns the python object which is owner _data allocation
    inline void *data_owner() {return _data_owner;}

    //! @brief returns the python object which is owner _indices allocation
    inline void *indices_owner() {return _indices_owner;}

    //! @brief returns the python object which is owner _row_indices allocation
    inline void *row_indices_owner() {return _row_indices_owner;}

    //! @brief Sets the data/indices allocation owner with either a numpy array or nullptr
    //! \param data A pointer to the data array
    //! \param indices A pointer to the indices array
    //! \param row_indices A pointer to the row_indices array
    //! \param n_rows The number of rows of the full array
    //! \param n_cols The number of columns of the full array
    //! \param data_owner A pointer to the numpy array (if nullptr, then the data allocation is
    //!         owned by the object itself) that owns _data allocation
    //! \param indices_owner A pointer to the numpy array (if nullptr, then the data allocation is
    //!         owned by the object itself) that owns _indices allocation
    //! \param row_indices_owner A pointer to the numpy array (if nullptr, then the data allocation
    //!         is owned by the object itself) that owns _row_indices allocation
    virtual void set_data_indices_rowindices(T *data, INDICE_TYPE *indices,
                                             INDICE_TYPE *row_indices, ulong n_rows, ulong n_cols,
                                             void *owner_data = nullptr,
                                             void *owner_indices = nullptr,
                                             void *owner_row_indices = nullptr);
#else
    //! @brief Sets the data. After this call, allocation is owned by the SArray2d.
    //! \param data A pointer to the data array
    //! \param indices A pointer to the indices array
    //! \param row_indices A pointer to the row_indices array
    //! \param n_rows The number of rows of the full array
    //! \param n_cols The number of columns of the full array
    virtual void set_data_indices_rowindices(T *data, INDICE_TYPE *indices,
                                             INDICE_TYPE *row_indices, ulong n_rows, ulong n_cols);
#endif

#ifdef DEBUG_SHAREDARRAY
    static ulong n_allocs;
#endif

 public:
    //
    // Constructors :
    //     One should only use the constructor
    //     SSparseArray2d<T>::new_ptr(n_rows, n_cols, size_sparse) which returns a shared pointer
    //     SSparseArray2d<T>::new_ptr(array) which returns a shared pointer
    //

    //! \cond
    // Constructor of an empty sparse array : not to be used directly
    // No allocation is performed whatever size is
    // Typically after this call one should call the set_data_indices_rowindices method.s
    explicit SSparseArray2d(ulong n_rows = 0, ulong n_cols = 0);
    //! \endcond

    //! @brief The only constructor to be used
    //! \return A shared pointer to the corresponding shared sparsearray of the given size.
    //! So if size_sparse != 0 then allocation is performed.
    //! \warning Right after this call, you are supposed to fill up the indices array with
    //! valid values
    static std::shared_ptr<SSparseArray2d<T>> new_ptr(ulong n_rows, ulong n_cols, ulong size_sparse);

    //! @brief The only constructor to be used from an SparseArray2d<T>
    //! \param a : The array the data/indices are copied from
    //! \return A shared pointer to the corresponding shared sparse array
    //! \warning The data/indices are copied
    static std::shared_ptr<SSparseArray2d<T>> new_ptr(SparseArray2d<T> &a);

 protected:
    /**
     * @brief Clears the array without desallocating the data pointer.
     * Returns this pointer if it should be desallocated otherwise returns
     * nullptr
     */
    virtual void _clear(bool &flag_desallocate_data, bool &flag_desallocate_indices,
                        bool &flag_desallocate_row_indices);

 public:
    //! @brief clears the corresponding allocation (n_rows and n_cols becomes 0)
    virtual void clear();

    //! @brief Destructor
    virtual ~SSparseArray2d();

    // The following constructors should not be used as we only use SSparseArray with the
    // associated shared_ptr
    SSparseArray2d(const SSparseArray2d<T> &other) = delete;
    SSparseArray2d(const SSparseArray2d<T> &&other) = delete;
};

#ifdef PYTHON_LINK
template <typename T>
void SSparseArray2d<T>::set_data_indices_rowindices(T *data, INDICE_TYPE *indices,
                                         INDICE_TYPE *row_indices, ulong n_rows, ulong n_cols,
                                         void *data_owner,
                                         void *indices_owner,
                                         void *row_indices_owner) {
    clear();
    _data = data;
    _indices = indices;
    _row_indices = row_indices;
    _size = n_rows*n_cols;
    _n_rows = n_rows;
    _n_cols = n_cols;
    _size_sparse = row_indices[n_rows];
    give_data_indices_rowindices_owners(data_owner, indices_owner, row_indices_owner);
}
#else
template<typename T>
void SSparseArray2d<T>::set_data_indices_rowindices(T *data, INDICE_TYPE *indices,
                                                    INDICE_TYPE *row_indices, ulong n_rows, ulong n_cols) {
    clear();
    _data = data;
    _indices = indices;
    _row_indices = row_indices;
    _size = n_rows * n_cols;
    _n_rows = n_rows;
    _n_cols = n_cols;
    _size_sparse = row_indices[n_rows];
    is_data_allocation_owned = true;
    is_row_indices_allocation_owned = true;
    is_indices_allocation_owned = true;
}
#endif

#ifdef PYTHON_LINK
template <typename T>
void SSparseArray2d<T>::give_data_indices_rowindices_owners(void *data_owner, void *indices_owner,
                                                          void *row_indices_owner) {
//    if (data_owner != nullptr || indices_owner != nullptr || row_indices_owner != nullptr)
//        THROWSTR("WEIRED give_data_indices_rowindices_owners");
#ifdef DEBUG_SHAREDARRAY
    if (_data_owner == nullptr) std::cout << "SSparseArray2d : SetOwner data_owner=" << data_owner << " on "
        <<  this << std::endl;
    else  std::cout << "SSparseArray2d : ChangeDataOwner data_owner=" << _data_owner <<" -> " << data_owner << " on "
        <<  this << std::endl;
    if (_indices_owner == nullptr) std::cout << "SSparseArray2d : SetOwner indices_owner=" << data_owner << " on "
        << this << std::endl;
    else  std::cout << "SSparseArray2d : ChangeIndicesOwner indices_owner=" << _indices_owner <<" -> " << indices_owner << " on "
        << this << std::endl;
    if (_row_indices_owner == nullptr) std::cout << "SSparseArray2d : SetOwner row_indices_owner=" << row_indices_owner << " on "
        << this << std::endl;
    else  std::cout << "SSparseArray2d : ChangeRowIndicesOwner row_indices_owner=" << _row_indices_owner <<" -> "<< row_indices_owner << " on "
        << this << std::endl;
#endif
    _data_owner = data_owner;
    if (_data_owner) {
        PYINCREF(_data_owner);
        is_data_allocation_owned = false;
    } else {
        is_data_allocation_owned = true;
    }

    _indices_owner = indices_owner;
    if (_indices_owner) {
        PYINCREF(_indices_owner);
        is_indices_allocation_owned = false;
    } else {
        is_indices_allocation_owned = true;
    }

    _row_indices_owner = row_indices_owner;
    if (_row_indices_owner) {
        PYINCREF(_row_indices_owner);
        is_row_indices_allocation_owned = false;
    } else {
        is_row_indices_allocation_owned = true;
    }
}
#endif

#ifdef DEBUG_SHAREDARRAY
template <typename T>
ulong SSparseArray2d<T>::n_allocs = 0;
#endif

// In order to create a Shared SparseArray
template<typename T>
SSparseArray2d<T>::SSparseArray2d(ulong n_rows, ulong n_cols) : SparseArray2d<T>() {
    _n_rows = n_rows;
    _n_cols = n_cols;
    _size = n_rows * n_cols;
#ifdef PYTHON_LINK
    _data_owner = nullptr;
    _indices_owner = nullptr;
    _row_indices_owner = nullptr;
#endif
#ifdef DEBUG_SHAREDARRAY
    n_allocs++;
    std::cout << "SSparseArray2d Constructor (->#" << n_allocs << ") : SSparseArray2d(n_rows=" << _n_rows << ",n_cols=" << _n_cols << ") --> "
    << this << std::endl;
#endif
}

// @brief The only constructor to be used
template<typename T>
std::shared_ptr<SSparseArray2d<T>> SSparseArray2d<T>::new_ptr(ulong n_rows, ulong n_cols, ulong size_sparse) {
    std::shared_ptr<SSparseArray2d<T>> aptr = std::make_shared<SSparseArray2d<T>>(n_rows, n_cols);
    if (n_rows == 0 || n_cols == 0 || size_sparse == 0) {
        return aptr;
    }
    T *data;
    TICK_PYTHON_MALLOC(data, T, size_sparse);
    INDICE_TYPE *indices;
    TICK_PYTHON_MALLOC(indices, INDICE_TYPE, size_sparse);
    INDICE_TYPE *row_indices;
    TICK_PYTHON_MALLOC(row_indices, INDICE_TYPE, n_rows + 1);
    row_indices[n_rows] = size_sparse;
    aptr->set_data_indices_rowindices(data, indices, row_indices, n_rows, n_cols);
    return aptr;
}

// Constructor from a SparseArray2d (copy is made)
template<typename T>
std::shared_ptr<SSparseArray2d<T>> SSparseArray2d<T>::new_ptr(SparseArray2d<T> &a) {
    std::shared_ptr<SSparseArray2d<T>> aptr = SSparseArray2d<T>::new_ptr(a.n_rows(), a.n_rows(), a.size_sparse());
    if (a.size_sparse() != 0) {
        memcpy(aptr->data(), a.data(), sizeof(T) * a.size_sparse());
        memcpy(aptr->indices(), a.indices(), sizeof(INDICE_TYPE) * a.size_sparse());
    }
    memcpy(aptr->row_indices(), a.row_indices(), sizeof(INDICE_TYPE) * (a.n_rows() + 1));
    return aptr;
}

// Clears the array without desallocation
// returns some flags to understand what should be desallocated
template<typename T>
void SSparseArray2d<T>::_clear(bool &flag_desallocate_data,
                               bool &flag_desallocate_indices,
                               bool &flag_desallocate_row_indices) {
    flag_desallocate_data = flag_desallocate_indices = flag_desallocate_row_indices = false;

    // Case not empty
    if (_data) {
#ifdef PYTHON_LINK
        // We have to deal with Python owner if any
        if (_data_owner != nullptr) {
            PYDECREF(_data_owner);
#ifdef DEBUG_SHAREDARRAY
            std::cout << "SSparseArray2d Clear : " << this
            << " decided not to free data=" << _data << " since owned by owner=" << _data_owner << std::endl;
#endif
            _data_owner = nullptr;
        } else {
#ifdef DEBUG_SHAREDARRAY
            std::cout << "SSparseArray2d Clear : " << this << " decided to free data=" << _data << std::endl;
#endif
            flag_desallocate_data = true;
        }
        if (_indices_owner != nullptr) {
            PYDECREF(_indices_owner);
#ifdef DEBUG_SHAREDARRAY
            std::cout << "SSparseArray2d Clear : " << this
            << " decided not to free indices=" << _indices << " since owned by owner=" << _indices_owner << std::endl;
#endif
            _indices_owner = nullptr;
        } else {
#ifdef DEBUG_SHAREDARRAY
            std::cout << "SSparseArray2d Clear : " << this << " decided to free indices="
            << _indices << std::endl;
            flag_desallocate_indices = true;
#endif
        }
#else
#ifdef DEBUG_SHAREDARRAY
        std::cout << "SSparseArray2d Clear : " << this << " decided to free data=" << _data << std::endl;
        std::cout << "SSparseArray2d Clear : " << this << " decided to free indices=" << _indices << std::endl;
#endif
        flag_desallocate_data = true;
        flag_desallocate_indices = true;
#endif
    }

    if (_row_indices) {
#ifdef PYTHON_LINK
        // We have to deal with Python owner if any
        if (_row_indices_owner != nullptr) {
            PYDECREF(_row_indices_owner);
#ifdef DEBUG_SHAREDARRAY
            std::cout << "SSparseArray2d Clear : " << this
            << " decided not to free row_indices=" << _row_indices << " since owned by owner=" << _row_indices_owner << std::endl;
#endif
            _row_indices_owner = nullptr;
        } else {
#ifdef DEBUG_SHAREDARRAY
            std::cout << "SSparseArray2d Clear : " << this << " decided to free data="
            << _row_indices_owner << std::endl;
#endif
            flag_desallocate_row_indices = true;
        }
#else
#ifdef DEBUG_SHAREDARRAY
        std::cout << "SSparseArray2d Clear : " << this << " decided to free row_indices=" << _row_indices << std::endl;
#endif
        flag_desallocate_row_indices = true;
#endif
    }

    _n_rows = 0;
    _n_cols = 0;
    _size = 0;
    _size_sparse = 0;

    is_row_indices_allocation_owned = true;
    is_indices_allocation_owned = true;
    is_data_allocation_owned = true;
}

// clear the array (including data if necessary)
template<typename T>
void SSparseArray2d<T>::clear() {
    bool flag_desallocate_data;
    bool flag_desallocate_indices;
    bool flag_desallocate_row_indices;
    _clear(flag_desallocate_data, flag_desallocate_indices, flag_desallocate_row_indices);
    if (flag_desallocate_data) TICK_PYTHON_FREE(_data);
    if (flag_desallocate_indices) TICK_PYTHON_FREE(_indices);
    if (flag_desallocate_row_indices) TICK_PYTHON_FREE(_row_indices);
    _data = nullptr;
    _indices = nullptr;
    _row_indices = nullptr;
}

// Destructor
template<typename T>
SSparseArray2d<T>::~SSparseArray2d() {
#ifdef DEBUG_SHAREDARRAY
    n_allocs--;
    std::cout << "SSparseArray2d Destructor (->#" << n_allocs << ") : ~SSparseArray2d on " <<  this << std::endl;
#endif
    clear();
}

// @brief Returns a shared pointer to a SSparseArray2d encapsulating the sparse array
// \warning : The ownership of the data is given to the returned structure
// THUS the array becomes a view.
// \warning : This method cannot be called on a view
template<typename T>
std::shared_ptr<SSparseArray2d<T>> SparseArray2d<T>::as_ssparsearray2d_ptr() {
    if (!is_data_allocation_owned || !is_indices_allocation_owned || !is_row_indices_allocation_owned)
        TICK_ERROR("This method cannot be called on an object that does not own its allocations");

    std::shared_ptr<SSparseArray2d<T>> arrayptr = SSparseArray2d<T>::new_ptr(0, 0, 0);
    arrayptr->set_data_indices_rowindices(_data, _indices, _row_indices, _n_rows, _n_cols);
    is_data_allocation_owned = false;
    is_indices_allocation_owned = false;
    is_row_indices_allocation_owned = false;
    return arrayptr;
}

// Instanciations

/**
 * \defgroup SArray_typedefs_mod SArray related typedef
 * \brief List of all the instantiations of the SArray template and associated
 *  shared pointers and 1d and 2d List of these classes
 * @{
 */

/**
 * @}
 */

/** @defgroup ssparsearray2d_sub_mod The instantiations of the SArray2d template
 *  @ingroup SArray_typedefs_mod
 * @{
 */

typedef SSparseArray2d<double> SSparseArrayDouble2d;
typedef SSparseArray2d<float> SSparseArrayFloat2d;
typedef SSparseArray2d<std::int32_t> SSparseArrayInt2d;
typedef SSparseArray2d<std::uint32_t> SSparseArrayUInt2d;
typedef SSparseArray2d<std::int16_t> SSparseArrayShort2d;
typedef SSparseArray2d<std::uint16_t> SSparseArrayUShort2d;
typedef SSparseArray2d<std::int64_t> SSparseArrayLong2d;
typedef SSparseArray2d<ulong> SSparseArrayULong2d;

/**
 * @}
 */

/** @defgroup ssparsearray2dptr_sub_mod The 2d shared pointer array classes
 *  @ingroup SArray_typedefs_mod
 * @{
 */

typedef std::shared_ptr<SSparseArrayFloat2d> SSparseArrayFloat2dPtr;
typedef std::shared_ptr<SSparseArrayInt2d> SSparseArrayInt2dPtr;
typedef std::shared_ptr<SSparseArrayUInt2d> SSparseArrayUInt2dPtr;
typedef std::shared_ptr<SSparseArrayShort2d> SSparseArrayShort2dPtr;
typedef std::shared_ptr<SSparseArrayUShort2d> SSparseArrayUShort2dPtr;
typedef std::shared_ptr<SSparseArrayLong2d> SSparseArrayLong2dPtr;
typedef std::shared_ptr<SSparseArrayULong2d> SSparseArrayULong2dPtr;
typedef std::shared_ptr<SSparseArrayDouble2d> SSparseArrayDouble2dPtr;

/**
 * @}
 */

/** @defgroup ssparsearray2dptrlist1d_sub_mod The classes for dealing with 1d-list of 2d shared pointer arrays
 *  @ingroup SArray_typedefs_mod
 * @{
 */

// @brief The basic SSparseArrayList1D classes
typedef std::vector<SSparseArrayFloat2dPtr> SSparseArrayFloat2dPtrList1D;
typedef std::vector<SSparseArrayInt2dPtr> SSparseArrayInt2dPtrList1D;
typedef std::vector<SSparseArrayUInt2dPtr> SSparseArrayUInt2dPtrList1D;
typedef std::vector<SSparseArrayShort2dPtr> SSparseArrayShort2dPtrList1D;
typedef std::vector<SSparseArrayUShort2dPtr> SSparseArrayUShort2dPtrList1D;
typedef std::vector<SSparseArrayLong2dPtr> SSparseArrayLong2dPtrList1D;
typedef std::vector<SSparseArrayULong2dPtr> SSparseArrayULong2dPtrList1D;
typedef std::vector<SSparseArrayDouble2dPtr> SSparseArrayDouble2dPtrList1D;

/**
 * @}
 */

/** @defgroup ssparsearray2dptrlist2d_sub_mod The classes for dealing with 2d-list of 2d shared pointer arrays
 *  @ingroup SSparseArray_typedefs_mod
 * @{
 */

// @brief The basic SSparseArrayList2D classes
typedef std::vector<SSparseArrayFloat2dPtrList1D> SSparseArrayFloat2dPtrList2D;
typedef std::vector<SSparseArrayInt2dPtrList1D> SSparseArrayInt2dPtrList2D;
typedef std::vector<SSparseArrayUInt2dPtrList1D> SSparseArrayUInt2dPtrList2D;
typedef std::vector<SSparseArrayShort2dPtrList1D> SSparseArrayShort2dPtrList2D;
typedef std::vector<SSparseArrayUShort2dPtrList1D> SSparseArrayUShort2dPtrList2D;
typedef std::vector<SSparseArrayLong2dPtrList1D> SSparseArrayLong2dPtrList2D;
typedef std::vector<SSparseArrayULong2dPtrList1D> SSparseArrayULong2dPtrList2D;
typedef std::vector<SSparseArrayDouble2dPtrList1D> SSparseArrayDouble2dPtrList2D;

/**
 * @}
 */

#endif  // TICK_BASE_ARRAY_SRC_SSPARSEARRAY2D_H_
