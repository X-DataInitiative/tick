#ifndef LIB_INCLUDE_TICK_ARRAY_VARRAY_H_
#define LIB_INCLUDE_TICK_ARRAY_VARRAY_H_

// License: BSD 3 clause

/** @file */

#include "sarray.h"

/*! \class VArray
 * \brief Template class for 1d non sparse arrays of type `T` fully shareable
 * with Python and in-place resizeable.
 *
 * So allocation can be owned either by Python interpreter or by
 * C++ and, in either case, can be seen in both environment (recall that `Array<T>`
 * allocated in C++ cannot be shared with python).
 * \warning You should always use the associated smart pointer class
 * `VArrayPtr` and never access this class directly. Thus the only constructor
 * to be used is the `VArray<T>::new_ptr()` constructor.
 *
 */
template<typename T>
class VArray : public SArray<T> {
 private:
    ulong alloc_size;
    static float factorIncrAlloc;

 protected :
    using SArray<T>::_size;
    using SArray<T>::_data;

 public :

#ifdef PYTHON_LINK
    //! @brief Sets the the data
    //! \param data A pointer to the data array
    //! \param size The size
    //! \param owner A pointer to owner (if nullptr then it is self-owned)
    virtual void set_data(T *ptr, ulong size, void *owner);
#else
    //! @brief Sets the data
    //! \param data A pointer to the data array
    //! \param size The size
    virtual void set_data(T *ptr, ulong size);
#endif

    /// @brief returns the preallocated size in case of extension
    ulong get_alloc_size() { return alloc_size; }

    //
    // Constructors :
    //     When not in a derived class, one should only use the constructor
    //     VArray<T>::new_ptr(size) which returns a shared pointer
    //     VArray<T>::new_ptr(array) which returns a shared pointer
    //

    //! \cond
    // Constructor of an array of size `size`
    explicit VArray(ulong size = 0);
    //! \endcond

    //! @brief The only constructor to be used
    //! \param size : the size of the array to be built
    //! \return A shared pointer to the corresponding shared array
    static std::shared_ptr<VArray<T>> new_ptr(ulong size = 0) {
        return std::make_shared<VArray<T>>(size);
    }

    //! @brief The only constructor to be used from an Array<T>
    //! \param a The array the data are copied from
    //! \return A shared pointer to the corresponding shared array
    //! \warning The data are copied
    static std::shared_ptr<VArray<T>> new_ptr(Array<T> &a);

    // These constructors should not be used as we only use VArray with the
    // associated shared_ptr
    VArray(const VArray<T> &other) = delete;
    VArray(const VArray<T> &&other) = delete;

 protected:
    virtual T *clear_();

 public:
    virtual ~VArray() {}

    /// @brief Manages allocation to get a specific size
    ///
    /// \param size The new size
    /// \param flagRemember If true, then former values are remembered
    virtual void set_size(ulong size, bool flagRemember = false);

    /// @brief Append one value at the end of the array
    /// \param value The value to be appended at the end of the array
    virtual void append1(T value);

    /// @brief Append another array of the same type at the end of the array
    /// \param sarray The array to be appended
    void append(std::shared_ptr<SArray<T>> sarray);
};

// Constructor
template<typename T>
std::shared_ptr<VArray<T>> VArray<T>::new_ptr(Array<T> &a) {
    if (a.size() == 0) return VArray<T>::new_ptr();
    std::shared_ptr<VArray<T>> aptr = VArray<T>::new_ptr(a.size());
    memcpy(aptr->data(), a.data(), sizeof(T) * a.size());
    return aptr;
}

/// @brief Factor by which the allowed memory will be increased when size
// reaches alloc_size.
#define VARRAY_FACTOR_INCRALLOC 1.5

template<typename T>
float VArray<T>::factorIncrAlloc = VARRAY_FACTOR_INCRALLOC;

template<typename T>
VArray<T>::VArray(ulong size) : SArray<T>(size) {
    alloc_size = size;
}

template<typename T>
void VArray<T>::set_size(ulong newsize, bool flagRemember) {
    if (newsize <= alloc_size) {
        Array<T>::_size = newsize;
        return;
    }
    ulong newSizeAlloc = VARRAY_FACTOR_INCRALLOC * newsize;
    T *tempPtr{nullptr};
#ifdef DEBUG_VARRAY
    std::cout << "VArray : SetSize : Alloc data of sizeAlloc=" << newSizeAlloc << " on " << this << std::endl;
#endif
    TICK_PYTHON_MALLOC(tempPtr, T, newSizeAlloc);
    if (flagRemember) memcpy(tempPtr, Array<T>::_data, sizeof(T) * Array<T>::_size);

    VArray<T>::clear();

    _data = tempPtr;
    alloc_size = newSizeAlloc;
    _size = newsize;
}

template<typename T>
void VArray<T>::append1(T value) {
    set_size(_size + 1, true);
    _data[_size - 1] = value;
}

template<typename T>
void VArray<T>::append(std::shared_ptr<SArray<T>> sarray) {
    ulong size_old = _size;
    ulong size_old1 = sarray->size();
    set_size(size_old + size_old1, true);
    memcpy(_data + size_old, sarray->data(), sizeof(T) * size_old1);
}

template<typename T>
T *VArray<T>::clear_() {
    alloc_size = 0;
    return SArray<T>::_clear();
}

#ifdef PYTHON_LINK
template<typename T>
void VArray<T>::set_data(T *ptr, ulong size, void *owner) {
    SArray<T>::set_data(ptr, size, owner);
    alloc_size = size;
}
#else
template<typename T>
void VArray<T>::set_data(T *ptr, ulong size) {
    SArray<T>::set_data(ptr, size);
    alloc_size = size;
}
#endif

/// @brief Allow us to easily print a VArray in standard output
template<typename T>
std::ostream &operator<<(std::ostream &Str, VArray<T> *v) {
#ifdef DEBUG_VARRAY
    Str << "VArray(" << reinterpret_cast<void *>(v) << ",size=" << v->size() << ")";
#else
    Str << "VArray(" << reinterpret_cast<void *>(v) << ",size=" << v->size() << ")";
#endif
    return Str;
}

// Instanciations

/**
 * \defgroup VArray_typedefs_mod VArray related typedef
 * \brief List of all the instantiations of the VArray template and associated
 *  shared pointers and 1d and 2d List of these classes
 * @{
 */

/**
 * @}
 */

/** @defgroup varray_sub_mod The instantiations of the VArray template
 *  @ingroup VArray_typedefs_mod
 * @{
 */

typedef VArray<double> VArrayDouble;
typedef VArray<float> VArrayFloat;
typedef VArray<std::int32_t> VArrayInt;
typedef VArray<std::uint32_t> VArrayUInt;
typedef VArray<std::int16_t> VArrayShort;
typedef VArray<std::uint16_t> VArrayUShort;
typedef VArray<std::int64_t> VArrayLong;
typedef VArray<ulong> VArrayULong;

/**
 * @}
 */

/** @defgroup varrayptr_sub_mod The shared pointer varray classes
 *  @ingroup VArray_typedefs_mod
 * @{
 */
typedef std::shared_ptr<VArrayFloat> VArrayFloatPtr;
typedef std::shared_ptr<VArrayInt> VArrayIntPtr;
typedef std::shared_ptr<VArrayUInt> VArrayUIntPtr;
typedef std::shared_ptr<VArrayShort> VArrayShortPtr;
typedef std::shared_ptr<VArrayUShort> VArrayUShortPtr;
typedef std::shared_ptr<VArrayLong> VArrayLongPtr;
typedef std::shared_ptr<VArrayULong> VArrayULongPtr;
typedef std::shared_ptr<VArrayDouble> VArrayDoublePtr;

/**
 * @}
 */

/** @defgroup varrayptrlist1d_sub_mod The classes for dealing with 1d-list of shared pointer varrays
 *  @ingroup VArray_typedefs_mod
 * @{
 */

typedef std::vector<VArrayFloatPtr> VArrayFloatPtrList1D;
typedef std::vector<VArrayIntPtr> VArrayIntPtrList1D;
typedef std::vector<VArrayUIntPtr> VArrayUIntPtrList1D;
typedef std::vector<VArrayShortPtr> VArrayShortPtrList1D;
typedef std::vector<VArrayUShortPtr> VArrayUShortPtrList1D;
typedef std::vector<VArrayLongPtr> VArrayLongPtrList1D;
typedef std::vector<VArrayULongPtr> VArrayULongPtrList1D;
typedef std::vector<VArrayDoublePtr> VArrayDoublePtrList1D;

/**
 * @}
 */

/** @defgroup varrayptrlist2d_sub_mod The classes for dealing with 2d-list of shared pointer varrays
 *  @ingroup VArray_typedefs_mod
 * @{
 */

// @brief The basic VArrayList2D classes
typedef std::vector<VArrayFloatPtrList1D> VArrayFloatPtrList2D;
typedef std::vector<VArrayIntPtrList1D> VArrayIntPtrList2D;
typedef std::vector<VArrayUIntPtrList1D> VArrayUIntPtrList2D;
typedef std::vector<VArrayShortPtrList1D> VArrayShortPtrList2D;
typedef std::vector<VArrayUShortPtrList1D> VArrayUShortPtrList2D;
typedef std::vector<VArrayLongPtrList1D> VArrayLongPtrList2D;
typedef std::vector<VArrayULongPtrList1D> VArrayULongPtrList2D;
typedef std::vector<VArrayDoublePtrList1D> VArrayDoublePtrList2D;

/** @}
 */

#define INSTANTIATE_VARRAY(VARRAY_TYPE, C_TYPE) \
template  std::ostream &  operator<< <C_TYPE>(std::ostream & Str, VArray<C_TYPE>  * v)

INSTANTIATE_VARRAY(VArrayDoublePtr, double);
INSTANTIATE_VARRAY(VArrayFloatPtr, float);
INSTANTIATE_VARRAY(VArrayIntPtr, std::int32_t);
INSTANTIATE_VARRAY(VArrayUIntPtr, std::uint32_t);
INSTANTIATE_VARRAY(VArrayShortPtr, std::int16_t);
INSTANTIATE_VARRAY(VArrayUShortPtr, std::uint16_t);
INSTANTIATE_VARRAY(VArrayLongPtr, std::int64_t);
INSTANTIATE_VARRAY(VArrayULongPtr, ulong);

#endif  // LIB_INCLUDE_TICK_ARRAY_VARRAY_H_
