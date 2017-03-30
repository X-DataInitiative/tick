//
//  time.h
//  Array
//
//  Created by bacry on 06/04/2015.
//  Copyright (c) 2015 bacry. All rights reserved.
//

#ifndef Array_time_h
#define Array_time_h

#include <time.h>

#include <iostream>
#include <iomanip>

#define START_TIMER(i, message) \
clock_t __timer ## i = clock(); \
std::string __timer_message ## i = message

#define END_TIMER(i) \
__timer ## i = clock() - __timer ## i; \
std::cout << __timer_message ## i << " : " << ((float)__timer ## i)/CLOCKS_PER_SEC << " seconds." << std::endl;

#define COMPARE_TIMER(i,j) \
float __timer_comp ## i ## j = (((float)__timer ## i) - ((float)__timer ## j))/((float)__timer ## j); \
if (__timer_comp ## i ## j >= 0) \
std::cout << __timer_message ## j << " is faster than " << __timer_message ## i << ", gain is : " << std::setprecision(3) << 100*__timer_comp ## i ## j << "%" << std::endl; \
else \
std::cout << __timer_message ## i << " is faster than " << __timer_message ## j << ", gain is : " << std::setprecision(3) << -100*__timer_comp ## i ## j << "%" << std::endl;
#endif