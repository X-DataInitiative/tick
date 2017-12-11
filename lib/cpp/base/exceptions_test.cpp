// License: BSD 3 clause

#include "tick/base/exceptions_test.h"

#include <stdexcept>
#include <system_error>
#include <cerrno>
#include <string>

void throw_out_of_range() {
    throw std::out_of_range("out_of_range");
}

void throw_system_error() {
    throw std::system_error(EACCES, std::generic_category());
}

void throw_invalid_argument() {
    throw std::invalid_argument("invalid_argument");
}

void throw_domain_error() {
    throw std::domain_error("domain_error");
}

void throw_runtime_error() {
    throw std::runtime_error("runtime_error");
}

void throw_string() {
    throw std::string("string");
}
