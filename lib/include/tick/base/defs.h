//
//  defs.h
//  TICK
//
//  Created by bacry on 27/12/2015.
//  Copyright (c) 2015 bacry. All rights reserved.
//
#ifndef LIB_INCLUDE_TICK_BASE_DEFS_H_
#define LIB_INCLUDE_TICK_BASE_DEFS_H_

// License: BSD 3 clause

#ifdef PYTHON_LINK
#include <Python.h>
#endif

#include <cstdint>

#if defined _WIN32 || defined __CYGWIN__
#ifdef BUILDING_DLL
#ifdef __GNUC__
#define DLL_PUBLIC __attribute__ ((dllexport))
#else
// Note: actually gcc seems to also supports this syntax.
#define DLL_PUBLIC __declspec(dllexport)
#endif
#else
#ifdef __GNUC__
#define DLL_PUBLIC __attribute__ ((dllimport))
#else
// Note: actually gcc seems to also supports this syntax.
#define DLL_PUBLIC __declspec(dllimport)
#endif
#endif
#define DLL_LOCAL
#else
#if __GNUC__ >= 4
#define DLL_PUBLIC __attribute__ ((visibility ("default")))
#define DLL_LOCAL  __attribute__ ((visibility ("hidden")))
#else
#define DLL_PUBLIC
#define DLL_LOCAL
#endif
#endif

#ifndef ulong
typedef std::uint64_t ulong;
#endif

#ifndef ushort
typedef std::uint16_t ushort;
#endif

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif  // _USE_MATH_DEFINES

#endif  // LIB_INCLUDE_TICK_BASE_DEFS_H_
