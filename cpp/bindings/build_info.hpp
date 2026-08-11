// bindings/build_info.hpp
//
// Lets Python see how an extension was compiled.
//
// Timings taken from an unoptimised build are off by roughly 5-6x on this
// project's inner loops — more than most of the speedups it sets out to
// measure, and entirely invisible in the output. Every benchmark script
// checks __optimized__ and says so rather than quietly printing a number that
// means nothing.

#pragma once

#include <pybind11/pybind11.h>

#ifndef HYLIS_BUILD_TYPE
#define HYLIS_BUILD_TYPE "unknown"
#endif

namespace hylis::bindings {

inline void attach_build_info(pybind11::module_& m) {
    m.attr("__build_type__") = HYLIS_BUILD_TYPE;
#ifdef NDEBUG
    m.attr("__optimized__") = true;
#else
    m.attr("__optimized__") = false;
#endif
}

}  // namespace hylis::bindings
