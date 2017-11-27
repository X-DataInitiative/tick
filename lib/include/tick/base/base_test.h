//
// Created by Martin Bompaire on 19/01/16.
//

#ifndef TICK_BASE_SRC_BASE_TEST_H_
#define TICK_BASE_SRC_BASE_TEST_H_

// License: BSD 3 clause

class A0 {
 protected:
    int cpp_int;

 public:
    int get_cpp_int() const {
        return cpp_int;
    }

    void set_cpp_int(int cpp_int) {
        this->cpp_int = cpp_int;
    }
};

#endif  // TICK_BASE_SRC_BASE_TEST_H_
