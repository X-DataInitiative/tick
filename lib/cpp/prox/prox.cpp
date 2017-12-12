// License: BSD 3 clause

#include "tick/prox/prox.h"

Prox::Prox(double strength,
           bool positive) {
  has_range = false;
  this->strength = strength;
  this->positive = positive;
}

Prox::Prox(double strength,
           ulong start,
           ulong end,
           bool positive) :
  Prox(strength, positive) {
  set_start_end(start, end);
}

const std::string Prox::get_class_name() const {
  return "Prox";
}

const bool Prox::is_separable() const {
  return false;
}

void Prox::call(const ArrayDouble &coeffs,
                double step,
                ArrayDouble &out) {
  if (has_range) {
    if (end > coeffs.size()) TICK_ERROR(
      get_class_name() << " of range [" << start << ", " << end
                       << "] cannot be called on a vector of size " << coeffs.size());
  } else {
    // If no range is given, we use the size of coeffs to get it
    set_start_end(0, coeffs.size());
    // But no range was given (set_start_end set has_range to true)
    // Still it is mandatory to set `start` and `end` has some prox might need to know if
    // they have changed from one iteration to another (see ProxSortedL1)
    has_range = false;
  }
  call(coeffs, step, out, start, end);
}

void Prox::call(const ArrayDouble &coeffs,
                double step,
                ArrayDouble &out,
                ulong start,
                ulong end) {
  TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

double Prox::value(const ArrayDouble &coeffs) {
  if (has_range) {
    if (end > coeffs.size()) TICK_ERROR(
      get_class_name() << " of range [" << start << ", " << end
                       << "] cannot get value of a vector of size " << coeffs.size());
  } else {
    // If no range is given, we use the size of coeffs to get it
    set_start_end(0, coeffs.size());
    // But no range was given (set_start_end set has_range to true)
    has_range = false;
  }
  return value(coeffs, start, end);
}

double Prox::value(const ArrayDouble &coeffs,
                   ulong start,
                   ulong end) {
  TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
}

double Prox::get_strength() const {
  return strength;
}

void Prox::set_strength(double strength) {
  this->strength = strength;
}

void Prox::set_start_end(ulong start,
                         ulong end) {
  if (start >= end) TICK_ERROR(
    get_class_name() << " can't have start(" << start
                     << ") greater or equal than end(" << end << ")");
  this->has_range = true;
  this->start = start;
  this->end = end;
}

ulong Prox::get_start() const {
  return start;
}

ulong Prox::get_end() const {
  return end;
}

bool Prox::get_positive() const {
  return positive;
}

void Prox::set_positive(bool positive) {
  this->positive = positive;
}
