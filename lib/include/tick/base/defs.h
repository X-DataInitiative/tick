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

#ifndef WIN32_MEAN_AND_LEAN
#define WIN32_MEAN_AND_LEAN
#endif  // WIN32_MEAN_AND_LEAN
#ifndef VC_EXTRA_LEAN
#define VC_EXTRA_LEAN
#endif  // VC_EXTRA_LEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX

#ifdef BUILDING_DLL
#ifdef __GNUC__
#define DLL_PUBLIC __attribute__((dllexport))
#else
// Note: actually gcc seems to also supports this syntax.
#define DLL_PUBLIC __declspec(dllexport)
#endif
#else
#ifdef __GNUC__
#define DLL_PUBLIC __attribute__((dllimport))
#else
// Note: actually gcc seems to also supports this syntax.
#define DLL_PUBLIC __declspec(dllimport)
#endif
#endif
#define DLL_LOCAL
#else
#if __GNUC__ >= 4
#define DLL_PUBLIC __attribute__((visibility("default")))
#define DLL_LOCAL __attribute__((visibility("hidden")))
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

// Lovingly stolen from https://stackoverflow.com/a/36175016/795574
#define DIAG_STR(s) #s
#define DIAG_JOINSTR(x, y) DIAG_STR(x##y)
#ifdef _MSC_VER
#define DIAG_DO_PRAGMA(x) __pragma(#x)
#define DIAG_PRAGMA(compiler, x) DIAG_DO_PRAGMA(warning(x))
#else
#define DIAG_DO_PRAGMA(x) _Pragma(#x)
#define DIAG_PRAGMA(compiler, x) DIAG_DO_PRAGMA(compiler diagnostic x)
#endif
#if defined(__clang__)
#define DISABLE_WARNING(gcc_unused, clang_option, msvc_unused) \
  DIAG_PRAGMA(clang, push)                                     \
  DIAG_PRAGMA(clang, ignored DIAG_JOINSTR(-W, clang_option))
#define ENABLE_WARNING(gcc_unused, clang_option, msvc_unused) \
  DIAG_PRAGMA(clang, pop)
#elif defined(_MSC_VER)
#define DISABLE_WARNING(gcc_unused, clang_unused, msvc_errorcode) \
  DIAG_PRAGMA(msvc, push) DIAG_DO_PRAGMA(warning(disable :##msvc_errorcode))
#define ENABLE_WARNING(gcc_unused, clang_unused, msvc_errorcode) \
  DIAG_PRAGMA(msvc, pop)
#elif defined(__GNUC__)
#if ((__GNUC__ * 100) + __GNUC_MINOR__) >= 406
#define DISABLE_WARNING(gcc_option, clang_unused, msvc_unused) \
  DIAG_PRAGMA(GCC, push) DIAG_PRAGMA(GCC, ignored DIAG_JOINSTR(-W, gcc_option))
#define ENABLE_WARNING(gcc_option, clang_unused, msvc_unused) \
  DIAG_PRAGMA(GCC, pop)
#else
#define DISABLE_WARNING(gcc_option, clang_unused, msvc_unused) \
  DIAG_PRAGMA(GCC, ignored DIAG_JOINSTR(-W, gcc_option))
#define ENABLE_WARNING(gcc_option, clang_option, msvc_unused) \
  DIAG_PRAGMA(GCC, warning DIAG_JOINSTR(-W, gcc_option))
#endif
#endif

// for function tracing and debugging activated with
// MKN_WITH=mkn.kul ./sh/mkn.sh
#ifdef _MKN_WITH_MKN_KUL_
#define KUL_FORCE_TRACE
#include "kul/dbg.hpp"
#else
#define KUL_DBG_FUNC_ENTER
#endif

#endif  // LIB_INCLUDE_TICK_BASE_DEFS_H_
