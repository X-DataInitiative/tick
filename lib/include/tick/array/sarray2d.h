//
//  sarray2d.h
//  tests
//
//  Created by bacry on 07/01/2016.
//  Copyright (c) 2016 bacry. All rights reserved.
//

#ifndef LIB_INCLUDE_TICK_ARRAY_SARRAY2D_H_
#define LIB_INCLUDE_TICK_ARRAY_SARRAY2D_H_

// License: BSD 3 clause

/** @file */

#include <memory>
#include "array2d.h"

///////////////////////////////////////////////////////////////////////////////////
//
//  SArray2d
//
///////////////////////////////////////////////////////////////////////////////////

// WARNING : An SArray should never have a C owner

/*! \class SArray2d
 * \brief Template class for 2d non sparse arrays of type `T` fully shareable
 * with Python.
 *
 * This is the array class that introduces the ability to share its allocation with Python.
 * \warning You should always use the associated smart pointer class
 * `SArray2dPtr` and never access this class directly. Thus the only constructor
 * to be used is the `SArray2d<T>::new_ptr()` constructor. In that way you can also share the
 * allocations within C++ (using a reference counter).
 */
template<typename T>
class SArray2d : public Array2d<T> {
 protected:
    using Array2d<T>::_size;
    using Array2d<T>::is_data_allocation_owned;
    using Array2d<T>::_data;
    using BaseArray2d<T>::_n_cols;
    using BaseArray2d<T>::_n_rows;

#ifdef PYTHON_LINK
    //! @brief The (eventual) Python owner of the array _data;
    //! If ==nullptr then it is self-owned
    void *_data_owner = nullptr;
#endif

 public:
#ifdef PYTHON_LINK
    void give_data_ownership(void *owner);

    //! @brief returns the python object which is owner _data allocation
    inline void *data_owner() {return _data_owner;}

    //! @brief Sets the data allocation owner with either a numpy array or nullptr
    //! \param data A pointer to the data array
    //! \param size The size
    //! \param owner A pointer to the numpy array (if nullptr, then the data allocation is owned by the object itself).
    virtual void set_data(T *data, ulong n_rows, ulong n_cols, void *owner = nullptr);
#else
    //! @brief Sets the data. After this call, allocation is owned by the SArray.
    //! \param data A pointer to the data array
    //! \param size The size
    virtual void set_data(T *data, ulong n_rows, ulong n_cols);
#endif

 protected:
#ifdef DEBUG_SHAREDARRAY
    static ulong n_allocs;
#endif

 public:
    // Constructors :
    //     One should only use the constructors
    //     SArray2d<T>::new_ptr(n_rows, n_cols) which returns a shared pointer
    //     SArray2d<T>::new_ptr(array2d) which returns a shared pointer
    //

    //! \cond
    // Constructor of an array of size `size` : not to be used directly
    // Allocation is performed if n_rows != 0 and n_cols != 0
    explicit SArray2d(ulong n_rows = 0, ulong n_cols = 0);
    //! \endcond

    //! @brief The only constructor to be used
    //! \param n_rows : the number of rows of the array to be built
    //! \param n_cols : the number of rows of the array to be built
    //! \return A shared pointer to the corresponding shared array of the given size
    //! Allcoation is performed
    static std::shared_ptr<SArray2d<T>> new_ptr(ulong n_rows = 0, ulong n_cols = 0) {
        return std::make_shared<SArray2d<T>>(n_rows, n_cols);
    }

    //! @brief The only constructor to be used from an Array2d<T>
    //! \param a : The array the data are copied from
    //! \return A shared pointer to the corresponding shared array
    //! \warning The data are copied
    static std::shared_ptr<SArray2d<T>> new_ptr(Array2d<T> &a);

 protected:
    /**
     * @brief clears the array without desallocating the data pointer.
     * Returns this pointer if it should be desallocated otherwise returns
     * nullptr
     */
    virtual T *_clear();

 public:
    //! @brief clears the corresponding allocation (size becomes 0)
    virtual void clear();

    //! @brief Destructor
    virtual ~SArray2d();

    // The following constructors should not be used as we only use SArray with the
    // associated shared_ptr
    SArray2d(const SArray2d<T> &other) = delete;
    SArray2d(const SArray2d<T> &&other) = delete;

    // Some friendship for set_data/give_ownership access !
    friend std::shared_ptr<SArray2d<T>> Array2d<T>::as_sarray2d_ptr();
};

#ifdef PYTHON_LINK
template <typename T>
void SArray2d<T>::set_data(T *data, ulong n_rows, ulong n_cols, void *nparray) {
    clear();
    _data = data;
    _n_cols = n_cols;
    _n_rows = n_rows;
    _size = n_cols*n_rows;
    give_data_ownership(nparray);
}
#else
template<typename T>
void SArray2d<T>::set_data(T *data, ulong n_rows, ulong n_cols) {
    clear();
    _data = data;
    _n_cols = n_cols;
    _n_rows = n_rows;
    _size = n_cols * n_rows;
    is_data_allocation_owned = true;
}
#endif

#ifdef PYTHON_LINK
template <typename T>
void SArray2d<T>::give_data_ownership(void *data_owner) {
#ifdef DEBUG_SHAREDARRAY
    if (_data_owner == nullptr) std::cout << "SArray2d : SetOwner owner=" << data_owner << " on "
        <<  this << std::endl;
    else  std::cout << "SArray2d : ChangeOwner owner=" << _data_owner << " -> " << data_owner << " on "
        << this << std::endl;
#endif
    _data_owner = data_owner;
    if (_data_owner) {
        PYINCREF(_data_owner);
        is_data_allocation_owned = false;
    } else {
        is_data_allocation_owned = true;
    }
}
#endif

#ifdef DEBUG_SHAREDARRAY
template <typename T>
ulong SArray2d<T>::n_allocs = 0;
#endif

// TODO : bof !
template<typename T>
std::ostream &operator<<(std::ostream &Str, SArray2d<T> *v) {
#ifdef DEBUG_SHAREDARRAY
    Str << "SArray2d(" << reinterpret_cast<void *>(v) << ", n_rows=" << v->n_rows() << ",n_cols=" << v->n_cols() << ")";
#else
    Str << "SArray2d(" << reinterpret_cast<void *>(v) << ",n_rows=" << v->n_rows() << ",n_cols=" << v->n_cols() << ")";
#endif
    return Str;
}

// In order to create a Shared array2d
template<typename T>
SArray2d<T>::SArray2d(ulong n_rows, ulong n_cols) : Array2d<T>(n_rows, n_cols) {
#ifdef PYTHON_LINK
    _data_owner = nullptr;
#endif
#ifdef DEBUG_SHAREDARRAY
    n_allocs++;
    std::cout << "SArray2d Constructor (->#" << n_allocs << ") : SArray2d(n_rows=" << _n_rows
    << ",n_cols=" << _n_cols << ") --> " << this << std::endl;
#endif
}

// Constructor from an Array2d (copy is made)
template<typename T>
std::shared_ptr<SArray2d<T>> SArray2d<T>::new_ptr(Array2d<T> &a) {
    if (a.size() == 0) return SArray2d<T>::new_ptr();
    std::shared_ptr<SArray2d<T>> aptr = SArray2d<T>::new_ptr(a.n_rows(), a.n_cols());
    memcpy(aptr->data(), a.data(), sizeof(T) * a.size());
    return aptr;
}

// clears the array2d without desallocating the data pointer.
// Returns this pointer if it should be desallocated otherwise returns NULL
// WARNING : An SArray2d should never have a C owner
template<typename T>
T *SArray2d<T>::_clear() {
    bool result = false;

    // Case not empty
    if (_data) {
#ifdef PYTHON_LINK
        // We have to deal with Python owner if any
        if (_data_owner != nullptr) {
            PYDECREF(_data_owner);
#ifdef DEBUG_SHAREDARRAY
            std::cout << "SArray2d Clear : " << this
                << " decided not to free data since owned by owner=" << _data_owner << std::endl;
#endif
            _data_owner = nullptr;
        } else {
#ifdef DEBUG_SHAREDARRAY
            std::cout << "SArray2d Clear : " << this << " decided to free data="
                << _data << std::endl;
#endif
            result = true;
        }
#else
#ifdef DEBUG_SHAREDARRAY
        std::cout << "SArray2d Clear : " << this << " decided to free data=" << _data << std::endl;
#endif
        result = true;
#endif
    }
    _size = 0;
    _n_rows = 0;
    _n_cols = 0;

    is_data_allocation_owned = true;

    return ((result ? _data : nullptr));
}

// clear the array (including data if necessary)
template<typename T>
void SArray2d<T>::clear() {
    if (_clear()) {
        TICK_PYTHON_FREE(_data);
    }
    _data = nullptr;
}

// Destructor
template<typename T>
SArray2d<T>::~SArray2d<T>() {
#ifdef DEBUG_SHAREDARRAY
    n_allocs--;
    std::cout << "SArray2d Destructor (->#" << n_allocs << ") : ~SArray2d on " <<  this << std::endl;
#endif
    clear();
}

// @brief Returns a shared pointer to a SArray2d encapsulating the array
// \warning : The ownership of the data is given to the returned structure
// THUS the array becomes a view.
// \warning : This method cannot be called on a view
template<typename T>
std::shared_ptr<SArray2d<T>> Array2d<T>::as_sarray2d_ptr() {
    if (!is_data_allocation_owned)
        TICK_ERROR("This method cannot be called on an object that does not own its allocations");

    std::shared_ptr<SArray2d<T>> arrayptr = SArray2d<T>::new_ptr();
    arrayptr->set_data(_data, _n_rows, _n_cols);
    is_data_allocation_owned = false;
    return arrayptr;
}

// Instantiations

/**
 * \defgroup SArray_typedefs_mod SArray related typedef
 * \brief List of all the instantiations of the SArray template and associated
 *  shared pointers and 1d and 2d List of these classes
 * @{
 */

/**
 * @}
 */

/** @defgroup sarray2d_sub_mod The instantiations of the SArray template
 *  @ingroup SArray2d_typedefs_mod
 * @{
 */

typedef SArray2d<double> SArrayDouble2d;
typedef SArray2d<float> SArrayFloat2d;
typedef SArray2d<std::int32_t> SArrayInt2d;
typedef SArray2d<std::uint32_t> SArrayUInt2d;
typedef SArray2d<std::int16_t> SArrayShort2d;
typedef SArray2d<std::uint16_t> SArrayUShort2d;
typedef SArray2d<std::int64_t> SArrayLong2d;
typedef SArray2d<std::uint64_t> SArrayULong2d;

/**
 * @}
 */

/** @defgroup sarray2dptr_sub_mod The shared pointer array classes
 *  @ingroup SArray2d_typedefs_mod
 * @{
 */

typedef std::shared_ptr<SArrayFloat2d> SArrayFloat2dPtr;
typedef std::shared_ptr<SArrayInt2d> SArrayInt2dPtr;
typedef std::shared_ptr<SArrayUInt2d> SArrayUInt2dPtr;
typedef std::shared_ptr<SArrayShort2d> SArrayShort2dPtr;
typedef std::shared_ptr<SArrayUShort2d> SArrayUShort2dPtr;
typedef std::shared_ptr<SArrayLong2d> SArrayLong2dPtr;
typedef std::shared_ptr<SArrayULong2d> SArrayULong2dPtr;
typedef std::shared_ptr<SArrayDouble2d> SArrayDouble2dPtr;

/**
 * @}
 */

/** @defgroup sarray2dptrlist1d_sub_mod The classes for dealing with 1d-list of shared pointer 2darrays
 *  @ingroup SArray_typedefs_mod
 * @{
 */

// @brief The basic SArray2dList1D classes
typedef std::vector<SArrayFloat2dPtr> SArrayFloat2dPtrList1D;
typedef std::vector<SArrayInt2dPtr> SArrayInt2dPtrList1D;
typedef std::vector<SArrayUInt2dPtr> SArrayUInt2dPtrList1D;
typedef std::vector<SArrayShort2dPtr> SArrayShort2dPtrList1D;
typedef std::vector<SArrayUShort2dPtr> SArrayUShort2dPtrList1D;
typedef std::vector<SArrayLong2dPtr> SArrayLong2dPtrList1D;
typedef std::vector<SArrayULong2dPtr> SArrayULong2dPtrList1D;
typedef std::vector<SArrayDouble2dPtr> SArrayDouble2dPtrList1D;

/**
 * @}
 */

/** @defgroup sarray2dptrlist2d_sub_mod The classes for dealing with 2d-list of shared pointer 2darrays
 *  @ingroup SArray2d_typedefs_mod
 * @{
 */

// @brief The basic SArray2dList2D classes
typedef std::vector<SArrayFloat2dPtrList1D> SArrayFloat2dPtrList2D;
typedef std::vector<SArrayInt2dPtrList1D> SArrayInt2dPtrList2D;
typedef std::vector<SArrayUInt2dPtrList1D> SArrayUInt2dPtrList2D;
typedef std::vector<SArrayShort2dPtrList1D> SArrayShort2dPtrList2D;
typedef std::vector<SArrayUShort2dPtrList1D> SArrayUShort2dPtrList2D;
typedef std::vector<SArrayLong2dPtrList1D> SArrayLong2dPtrList2D;
typedef std::vector<SArrayULong2dPtrList1D> SArrayULong2dPtrList2D;
typedef std::vector<SArrayDouble2dPtrList1D> SArrayDouble2dPtrList2D;

/**
 * @}
 */

#define INSTANTIATE_SARRAY2D(SARRAY_TYPE, C_TYPE) \
template std::ostream& operator<< <C_TYPE>(std::ostream & Str, SArray2d<C_TYPE>  * v)

INSTANTIATE_SARRAY2D(SArrayDouble2dPtr, double);
INSTANTIATE_SARRAY2D(SArrayFloat2dPtr, float);
INSTANTIATE_SARRAY2D(SArrayInt2dPtr, std::int32_t);
INSTANTIATE_SARRAY2D(SArrayUInt2dPtr, std::uint32_t);
INSTANTIATE_SARRAY2D(SArrayShort2dPtr, std::int16_t);
INSTANTIATE_SARRAY2D(SArrayUShort2dPtr, std::uint16_t);
INSTANTIATE_SARRAY2D(SArrayLong2dPtr, std::int64_t);
INSTANTIATE_SARRAY2D(SArrayULong2dPtr, ulong);

#endif  // LIB_INCLUDE_TICK_ARRAY_SARRAY2D_H_
