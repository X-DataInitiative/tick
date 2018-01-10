//
//  time.h
//  Array
//
//  Created by bacry on 06/04/2015.
//  Copyright (c) 2015 bacry. All rights reserved.
//

#ifndef LIB_INCLUDE_TICK_ARRAY_TEST_TIMER_H_
#define LIB_INCLUDE_TICK_ARRAY_TEST_TIMER_H_

// License: BSD 3 clause

#include "tick/base/defs.h"

#include <iostream>
#include <sstream>
#include <iomanip>

#include <ctime>

#define START_TIMER(i, message) \
clock_t __timer ## i = clock(); std::string __timer_message ## i = message

#define END_TIMER(i) \
__timer ## i = clock() - __timer ## i; \
std::cout << __timer_message ## i << " : " << (static_cast<float>(__timer ## i))/CLOCKS_PER_SEC << " seconds." << std::endl;

namespace tick {

inline void compare_timers(const clock_t& t1, const std::string& msg1, const clock_t& t2, const std::string& msg2) {
    const float f11 = static_cast<float>(t1);
    const float f12 = static_cast<float>(t2);

    const float comp = (f11 - f12) / f12;

    if (comp >= 0.0f) {
        std::cout << msg2 << " is faster than " << msg1 << ", gain is: " << std::setprecision(3) <<  100.0f * comp << "%" << std::endl;
    } else {
        std::cout << msg1 << " is faster than " << msg2 << ", gain is: " << std::setprecision(3) << -100.0f * comp << "%" << std::endl;
    }
}

}  // namespace tick

#define COMPARE_TIMER(i, j) tick::compare_timers(__timer ## i, __timer_message ## i , __timer ## j, __timer_message ## j)

#endif  // LIB_INCLUDE_TICK_ARRAY_TEST_TIMER_H_
