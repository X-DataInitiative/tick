// License: BSD 3 clause

//
// Created by Martin Bompaire on 08/06/15.
//

#include "tick/base/time_func.h"
#include <float.h>
#define FLOOR_THRESHOLD 1e-10

TimeFunction::TimeFunction(double y) {
  // Little trick so that it will predict y after 0 and 0 before
  // 2* is needed to ensure that 0 > last_value_before_border + FLOOR_THRESHOLD
  last_value_before_border = -2 * FLOOR_THRESHOLD;
  border_value = y;
  border_type = DEFAULT_BORDER;
}

TimeFunction::TimeFunction(const ArrayDouble &Y, BorderType type, InterMode mode, double dt,
                           double border_value)
    : inter_mode(mode), border_type(type), dt(dt), border_value(border_value) {
  set_t0(0.);  // Implicitly we are assuming that support starts at 0
  ulong size = Y.size();
  sampled_y = SArrayDouble::new_ptr(size);
  std::copy(Y.data(), Y.data() + size, sampled_y->data());

  sampled_y_primitive = SArrayDouble::new_ptr(size);
  for (ulong i = 0; i < size; ++i) {
    if (i == 0) {
      (*sampled_y_primitive)[i] = 0;
    } else {
      double y_i_1 = (*sampled_y)[i - 1];
      double y_i = (*sampled_y)[i];
      double y;
      switch (inter_mode) {
        case (InterMode::InterConstLeft): {
          y = y_i;
          break;
        }
        case (InterMode::InterConstRight): {
          y = y_i_1;
          break;
        }
        case (InterMode::InterLinear): {
          y = (y_i + y_i_1) / 2.;
          break;
        }
      }
      (*sampled_y_primitive)[i] = (*sampled_y_primitive)[i - 1] + y * dt;
    }
  }

  last_value_before_border = dt * sampled_y->size();
}

TimeFunction::TimeFunction(const ArrayDouble &T, const ArrayDouble &Y, double dt)
    : TimeFunction(T, Y, DEFAULT_BORDER, DEFAULT_INTER, dt, 0.0) {}

TimeFunction::TimeFunction(const ArrayDouble &T, const ArrayDouble &Y, BorderType type,
                           InterMode mode, double dt1, double border_value1)
    : inter_mode(mode), border_type(type) {
  const ulong size = T.size();
  if (size != Y.size()) TICK_ERROR("Both array should have the same size");
  if (size == 0) TICK_ERROR("T and Y arrays cannot be empty");
  if (dt1 < 0) TICK_ERROR("``dt` cannot be negative");
  if (T[0] < 0)
    TICK_ERROR(
        "TimeFunction support is on R+, hence no negative time can be "
        "provided");

  // Check that T array is sorted and find min step
  double min_step = DBL_MAX;
  for (ulong i = 1; i < size; ++i) {
    double step = T[i] - T[i - 1];
    if (step < -FLOOR_THRESHOLD) TICK_ERROR("T-array must be sorted");
    if (step < FLOOR_THRESHOLD) continue;
    if (step < min_step) min_step = step;
  }

  // Specify dt if user has not specified it
  if (dt1 == 0) {
    dt = min_step;
  } else {
    dt = dt1;
  }

  if (dt < 10 * FLOOR_THRESHOLD)
    TICK_ERROR("dt is too small, we currently cannot reach this precision");

  set_t0(T[0]);

  last_value_before_border = T[size - 1];
  switch (border_type) {
    case (BorderType::Border0):
      border_value = 0;
      break;
    case (BorderType::BorderConstant):
      border_value = border_value1;
      break;
    case (BorderType::BorderContinue):
      border_value = Y[size - 1];
      break;
    case (BorderType::Cyclic):
      border_value = 0;
      break;
  }

  // two cases
  // either dt divides length (for example T = [1, 2, 3] and dt = 0.5)
  // either dt don't (common case)
  const double length = T[size - 1] - T[0];

  if (length < dt) TICK_ERROR("``dt`` cannot be bigger than the time space you are considering");

  ulong sample_size = 1 + std::ceil(length / dt);

  sampled_y = SArrayDouble::new_ptr(sample_size);
  sampled_y_primitive = SArrayDouble::new_ptr(sample_size);

  // We iterate over all given points and evaluate the function between them to
  // fill sampled_y by dt step
  double t = t0;

  // Index of the left point selected
  ulong index_left = 0;
  // initialize with the first two points
  double t_left = T[index_left];
  double y_left = Y[index_left];
  double t_right = T[index_left + 1];
  double y_right = Y[index_left + 1];

  for (ulong i = 0; i < sampled_y->size(); ++i) {
    while (t > t_right && index_left < size - 2) {
      index_left += 1;
      t_left = T[index_left];
      y_left = Y[index_left];
      t_right = T[index_left + 1];
      y_right = Y[index_left + 1];
    }
    double y_i;
    if (t_left + FLOOR_THRESHOLD < t) {
      if (t < t_right - FLOOR_THRESHOLD)
        y_i = interpolation(t_left, y_left, t_right, y_right, t);
      else
        y_i = y_right;
    } else
      y_i = y_left;
    /*
    std::cout << "\nTimeFunction::TimeFunction  " << std::endl
              << " i = " << i << ", " << std::endl
              << " index_left = " << index_left << ", " << std::endl
              << " t_left: " << t_left << std::endl
              << " t: " << t << std::endl
              << " t_right: " << t_right << std::endl
              << " y_left: " << y_left << std::endl
              << " y_i: " << y_i << std::endl
              << " y_right: " << y_right << std::endl
              << std::endl;
              */
    (*sampled_y)[i] = y_i;
    if (i == 0)
      (*sampled_y_primitive)[0] = 0;
    else {
      double y_i_1 = (*sampled_y)[i - 1];
      double y;
      switch (inter_mode) {
        case (InterMode::InterConstLeft): {
          y = y_i;
          break;
        }
        case (InterMode::InterConstRight): {
          y = y_i_1;
          break;
        }
        case (InterMode::InterLinear): {
          y = (y_i + y_i_1) / 2.;
          break;
        }
      }
      (*sampled_y_primitive)[i] = (*sampled_y_primitive)[i - 1] + y * dt;
    }
    t += dt;
  }

  support_right = sampled_y->size() * dt + t0;
}

TimeFunction::~TimeFunction() {
  if (sampled_y) sampled_y.reset();
  if (sampled_y_primitive) sampled_y_primitive.reset();
  if (future_max) future_max.reset();
}

void TimeFunction::compute_future_max() {
  // Check this TimeFunction is not constant
  if (last_value_before_border < 0) return;

  ulong sample_size = sampled_y->size();
  future_max = SArrayDouble::new_ptr(sample_size);

  if (border_type != BorderType::Cyclic) {
    double previous_max = border_value;
    // condition is : i + 1 > 0 as an ulong cannot be negative
    for (ulong i = sample_size - 1; i + 1 > 0; --i) {
      (*future_max)[i] = std::max((*sampled_y)[i], previous_max);
      previous_max = (*future_max)[i];
    }
  } else {
    // if border is cyclic then future max in global max
    double max_value = sampled_y->max();
    future_max->fill(max_value);
  }
}

double TimeFunction::value(double t) {
  // First handle if we are out of the border

  // this test must be the first one otherwise we might have problem with
  // constant TimeFunctions
  if (t > last_value_before_border + FLOOR_THRESHOLD) {
    if (border_type != BorderType::Cyclic) {
      return border_value;
    } else {
      // If border type is cyclic then we simply return the value it would have
      // in the first cycle
      const double divider = last_value_before_border;
      const int quotient = static_cast<int>(std::ceil(t / divider));
      return value(t - quotient * divider);
    }
  } else if (t < t0) {
    // which behavior do we want if t < t0 ?
    return 0.0;
  } else if (t < 0) {
    return 0.0;
  }

  const ulong i_left = _idx_left(t);
  const ulong i_right = _idx_right(t);
  const double t_left = _t_left(t);
  const double t_right = _t_right(t);
  const double y_left = (*sampled_y)[i_left];
  const double y_right = (*sampled_y)[i_right];

  return interpolation(t_left, y_left, t_right, y_right, t);
}

double TimeFunction::primitive(double t) {
  if (t > last_value_before_border + FLOOR_THRESHOLD) {
    if (border_type != BorderType::Cyclic) {
      double delay = t - last_value_before_border;
      return primitive(last_value_before_border) + border_value * delay;
    } else {
      const double divider = last_value_before_border;
      const int quotient = static_cast<int>(std::ceil(t / divider));
      return quotient * primitive(last_value_before_border) + primitive(t - quotient * divider);
    }
  } else if (t < t0) {
    // which behavior do we want if t < t0 ?
    return 0.0;
  } else if (t < 0) {
    return 0.0;
  }

  const ulong i_left = get_index_(t);

  const double t_left = get_t_from_index_(i_left);
  const double y_left = (*sampled_y_primitive)[i_left];
  const double t_right = get_t_from_index_(i_left + 1);
  const double y_right = (*sampled_y_primitive)[i_left + 1];
  const double slope = (y_right - y_left) / (t_right - t_left);
  return y_left + (t - t_left) * slope;
}

SArrayDoublePtr TimeFunction::value(ArrayDouble &array) {
  SArrayDoublePtr value_array = SArrayDouble::new_ptr(array.size());
  for (ulong i = 0; i < value_array->size(); ++i) {
    (*value_array)[i] = value(array[i]);
  }
  return value_array;
}

double TimeFunction::future_bound(double t) {
  if (future_max == nullptr) {
    compute_future_max();
  }

  if (t > last_value_before_border) {
    // First handle if we are out of the border
    if (border_type != BorderType::Cyclic) {
      return border_value;
    } else {
      // If border type is cyclic then we simply return the value it would have
      // in the first cycle
      const double divider = last_value_before_border;
      const int quotient = static_cast<int>(std::ceil(t / divider));
      return future_bound(t - quotient * divider);
    }
  } else if (t < t0) {
    return (*future_max)[0];
  } else if (t < 0) {
    return (*future_max)[0];
  }

  const ulong i_left = get_index_(t);

  const double t_left = get_t_from_index_(i_left);
  const double y_left = (*future_max)[i_left];
  const double t_right = get_t_from_index_(i_left + 1);
  const double y_right = (*future_max)[i_left + 1];

  return interpolation(t_left, y_left, t_right, y_right, t);
}

SArrayDoublePtr TimeFunction::future_bound(ArrayDouble &array) {
  SArrayDoublePtr future_max_array = SArrayDouble::new_ptr(array.size());
  for (ulong i = 0; i < future_max_array->size(); ++i) {
    (*future_max_array)[i] = future_bound(array[i]);
  }
  return future_max_array;
}

double TimeFunction::max_error(double t) {
  const ulong i_left = get_index_(t);

  const double t_left = get_t_from_index_(i_left);
  const double y_left = (*sampled_y)[i_left];
  const double t_right = get_t_from_index_(i_left + 1);
  const double y_right = (*sampled_y)[i_left + 1];

  switch (inter_mode) {
    case (InterMode::InterLinear):
      if (std::abs(t_left - t_right) < FLOOR_THRESHOLD) {
        return std::abs(y_left - y_right);
      } else {
        const double slope = (y_right - y_left) / (t_right - t_left);
        return std::abs(slope) * dt;
      }
    case (InterMode::InterConstLeft):
      return FLOOR_THRESHOLD;
    case (InterMode::InterConstRight):
      return FLOOR_THRESHOLD;
    default:
      return 0;
  }
}

double TimeFunction::get_norm() {
  // Check this TimeFunction is not constant
  if (last_value_before_border < 0) {
    if (border_value == 0)
      return 0;
    else
      return DBL_MAX;
  }

  double norm = 0;
  for (ulong i = 1; i < sampled_y->size(); ++i) {
    double y_left = (*sampled_y)[i - 1];
    double y_right = (*sampled_y)[i];

    switch (inter_mode) {
      case (InterMode::InterLinear):
        norm += .5 * (y_left + y_right) * dt;
        break;
      case (InterMode::InterConstLeft):
        // only right point matters
        norm += y_right * dt;
        break;
      case (InterMode::InterConstRight):
        // only left point matters
        norm += y_left * dt;
        break;
    }
  }
  return norm;
}

double TimeFunction::constant_left_interpolation(double t_left, double y_left, double t_right,
                                                 double y_right, double t_value) {
  if (std::abs(t_value - t_left) < FLOOR_THRESHOLD)
    return y_left;
  else
    return y_right;
}

double TimeFunction::constant_right_interpolation(double t_left, double y_left, double t_right,
                                                  double y_right, double t_value) {
  if (std::abs(t_value - t_right) < FLOOR_THRESHOLD)
    return y_right;
  else
    return y_left;
}

double TimeFunction::linear_interpolation(double t_left, double y_left, double t_right,
                                          double y_right, double t_value) {
  if (std::abs(t_left - t_right) < FLOOR_THRESHOLD) {
    return (y_left + y_right) / 2.0;
  } else {
    double slope = (y_right - y_left) / (t_right - t_left);
    return y_left + slope * (t_value - t_left);
  }
}

double TimeFunction::interpolation(double t_left, double y_left, double t_right, double y_right,
                                   double t) {
  if (t < t_left)
    throw std::runtime_error("TimeFunction::interpolation error: t_left cannot be larger than t");
  if (t > t_right)
    throw std::runtime_error("TimeFunction::interpolation error: t_right cannot be smaller than t");
  // Second do the right interpolation
  switch (inter_mode) {
    case (InterMode::InterLinear):
      return linear_interpolation(t_left, y_left, t_right, y_right, t);
    case (InterMode::InterConstLeft):
      return constant_left_interpolation(t_left, y_left, t_right, y_right, t);
    case (InterMode::InterConstRight):
      return constant_right_interpolation(t_left, y_left, t_right, y_right, t);
    default:
      throw std::runtime_error("Undefined interpolation mode");
  }
}

ulong TimeFunction::get_index_(double t) { return (ulong)std::ceil((t - t0) / dt); }

double TimeFunction::get_t_from_index_(ulong i) { return t0 + dt * i; }

ulong TimeFunction::_idx_left(double t) {
  ulong k = std::floor((t - t0) / dt);
  return k;
}

ulong TimeFunction::_idx_right(double t) {
  ulong k = std::ceil((t - t0 + FLOOR_THRESHOLD) / dt);
  return k;
}

double TimeFunction::_t_left(double t) {
  ulong k = std::floor((t - t0) / dt);
  return t0 + k * dt;
}

double TimeFunction::_t_right(double t) {
  ulong k = std::ceil((t - t0 + FLOOR_THRESHOLD) / dt);
  return t0 + k * dt;
}
