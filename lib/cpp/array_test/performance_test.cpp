// License: BSD 3 clause

//
// Created by Martin Bompaire on 22/05/15.
//

#include "tick/array_test/performance_test.h"

using std::cout;
using std::endl;

double test_sum_double_pointer(ulong size, ulong n_loops) {
    double *values = new double[size];
    for (ulong i = 0; i < size; i++) values[i] = 1;

    double sum = 0;
    for (ulong j = 0; j < n_loops; j++) {
        for (ulong i = 0; i < size; i++)
            sum += values[i];
    }
    delete[] values;
    return sum;
}

double test_sum_ArrayDouble(ulong size, ulong n_loops) {
    ArrayDouble array = ArrayDouble(size);
    array.fill(1);

    double sum = 0;
    for (ulong j = 0; j < n_loops; j++) {
        for (ulong i = 0; i < size; i++)
            sum += array[i];
    }
    return sum;
}

double test_sum_SArray_shared_ptr(ulong size, ulong n_loops) {
    SArrayDoublePtr sarray = SArrayDouble::new_ptr(size);
    sarray->fill(1);

    double sum = 0;
    for (ulong j = 0; j < n_loops; j++) {
        for (ulong i = 0; i < size; i++)
            sum += (*sarray)[i];
    }
    return sum;
}

double test_sum_VArray_shared_ptr(ulong size, ulong n_loops) {
    VArrayDoublePtr varray = VArrayDouble::new_ptr(size);
    varray->fill(1);

    double sum = 0;
    for (ulong j = 0; j < n_loops; j++) {
        for (ulong i = 0; i < size; i++)
            sum += (*varray)[i];
    }
    return sum;
}

// Testing array_double accessing directly to the pointer
// versus the inline [] method
void test_element_access() {
    ulong size = 50000;
    ulong nLoops = 10000;

    cout << "*** Testing accessing elements of ArrayDouble and VArrayDouble" << endl;

    double *values = new double[size];
    for (ulong i = 0; i < size; i++) values[i] = 1;
    ArrayDouble array = ArrayDouble(size, values);

    array.print();
    double sum = 0;
    double *val = array.data();

    START_TIMER(1, "fast_array");
    for (ulong j = 0; j < nLoops; j++) {
        for (ulong i = 0; i < array.size(); i++)
            sum += val[i];
    }
    END_TIMER(1);
    cout << "sum = " << sum << endl;

    sum = 0;
    START_TIMER(2, "slow_array");
    for (ulong j = 0; j < nLoops; j++) {
        for (ulong i = 0; i < array.size(); i++)
            sum += array[i];
    }
    END_TIMER(2);
    cout << "sum = " << sum << endl;



    cout << "-------------" << endl;
    cout << "intial data=" << values << endl;
    sum = 0;
    VArrayDoublePtr vv = VArrayDouble::new_ptr(size);
    for (ulong i = 0; i < size; i++) (*vv)[i] = values[i];

    cout << "created " << vv << endl;

    START_TIMER(4, "varray_shared_ptr");
    for (ulong j = 0; j < nLoops; j++) {
        for (ulong i = 0; i < vv->size(); i++)
            sum += (*vv)[0];
    }
    END_TIMER(4);
    cout << "sum =  " << sum << endl;


    COMPARE_TIMER(1, 2);
    COMPARE_TIMER(1, 4);

    delete[] values;
}

double inherited_func(InheritedArray* array, ulong* nLoops) {
    double sum = 0;
    for (ulong j = 0; j < *nLoops; j++) {
        for (ulong i = 0; i < array->size; i++)
            sum += (*array)[i];
    }
    return sum;
}

double abstract_func(ToyAbstractArray * array, ulong* nLoops) {
    double sum = 0.0;
    for (ulong j = 0; j < *nLoops; j++) {
        for (ulong i = 0; i < array->size; i++)
            sum += array->getValue(i);
    }
    return sum;
}

double inherited_func_no_ptr(InheritedArray array, ulong nLoops) {
    double sum = 0.0;
    for (ulong j = 0; j < nLoops; j++) {
        for (ulong i = 0; i < array.size; i++)
            sum += array[i];
    }
    return sum;
}


// Testing array_double accessing directly to the pointer
// versus the inline [] method
void test_element_access_inherited_array() {
    ulong size = 50000;
    ulong nLoops = 10000;

    cout << "\n*** Testing accessing elements from inherited Array " << endl;

    double *values = new double[size];
    for (ulong i = 0; i < size; i++) values[i] = 1;
    ArrayDouble array = ArrayDouble(size, values);

    double sum = 0;

    START_TIMER(1, "direct_array");
    for (ulong j = 0; j < nLoops; j++) {
        for (ulong i = 0; i < array.size(); i++)
            sum += array[i];
    }
    END_TIMER(1);
    cout << "sum = " << sum << endl;


    InheritedArray inheritedArray = InheritedArray(values, size);
    InheritedArray* inheritedArrayPtr = &inheritedArray;
    ToyAbstractArray * abstractArrayPtr = &inheritedArray;

    sum = 0;
    START_TIMER(5, "inherited_array");
    for (ulong j = 0; j < nLoops; j++) {
        for (ulong i = 0; i < inheritedArray.size; i++)
            sum += inheritedArray[i];
    }
    END_TIMER(5);
    cout << "sum = " << sum << endl;

    sum = 0;
    START_TIMER(6, "inherited_array_in_function");
    sum = inherited_func_no_ptr(inheritedArray, nLoops);
    END_TIMER(6);
    cout << "sum = " << sum << endl;

    sum = 0;
    START_TIMER(2, "inherited_array_ptr");
    for (ulong j = 0; j < nLoops; j++) {
        for (ulong i = 0; i < inheritedArrayPtr->size; i++)
            sum += (*inheritedArrayPtr)[i];
    }
    END_TIMER(2);
    cout << "sum = " << sum << endl;

    sum = 0;
    START_TIMER(8, "abstract_array_ptr");
    for (ulong j = 0; j < nLoops; j++) {
        for (ulong i = 0; i < abstractArrayPtr->size; i++)
            sum += abstractArrayPtr->getValue(i);
    }
    END_TIMER(8);
    cout << "sum = " << sum << endl;

    START_TIMER(3, "inherited_array_in_function_ptr");
    sum = inherited_func(&inheritedArray, &nLoops);
    END_TIMER(3);
    cout << "sum = " << sum << endl;


    START_TIMER(4, "abstract_array_in_function_ptr");
    sum = abstract_func(&inheritedArray, &nLoops);
    END_TIMER(4);
    cout << "sum = " << sum << endl;


    COMPARE_TIMER(1, 5);
    COMPARE_TIMER(1, 6);
    COMPARE_TIMER(5, 2);
    COMPARE_TIMER(2, 3);
    COMPARE_TIMER(2, 4);


    delete[] values;
}
