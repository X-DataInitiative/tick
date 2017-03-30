//
// Created by Martin Bompaire on 08/06/15.
//

#include <float.h>
#include "time_func.h"

#define FLOOR_THRESHOLD 1e-10
#define FLOOR(x) std::floor((x) + FLOOR_THRESHOLD)

#define TIME_FUNC_OVERSAMPLING 5.0

TimeFunction::TimeFunction(double y) {
  // Little trick so that it will predict y after 0 and 0 before
  // 2* is needed to ensure that 0 > last_value_before_border + FLOOR_THRESHOLD
  last_value_before_border = -2 * FLOOR_THRESHOLD;
  border_value = y;
  border_type = DEFAULT_BORDER;
}

TimeFunction::TimeFunction(const ArrayDouble &Y, BorderType type, InterMode mode, double dt, double border_value)
    : inter_mode(mode), border_type(type), dt(dt), border_value(border_value) {
  sampled_y = SArrayDouble::new_ptr(Y.size());
  std::copy(Y.data(), Y.data() + Y.size(), sampled_y->data());

  last_value_before_border = dt * sampled_y->size();
}

TimeFunction::TimeFunction(const ArrayDouble &T, const ArrayDouble &Y, double dt)
    : TimeFunction(T, Y, DEFAULT_BORDER, DEFAULT_INTER, dt, 0.0) {
}

TimeFunction::TimeFunction(const ArrayDouble &T, const ArrayDouble &Y,
                           BorderType type, InterMode mode, double dt1, double border_value1)
    : inter_mode(mode), border_type(type) {

  const ulong size = T.size();
  if (size != Y.size()) TICK_ERROR("Both array should have the same size");
  if (size == 0) TICK_ERROR("T and Y arrays cannot be empty");
  if (dt1 < 0) TICK_ERROR("``dt` cannot be negative");
  if (T[0] < 0) TICK_ERROR("TimeFunction support is on R+, hence no negative time can be provided");

  // Check that T array is sorted and find min step
  double min_step = DBL_MAX;
  for (ulong i = 1; i < size; ++i) {
    double step = T[i] - T[i - 1];
    if (step < -FLOOR_THRESHOLD) TICK_ERROR("X-array must be sorted");
    if (step < FLOOR_THRESHOLD) continue;
    if (step < min_step) min_step = step;
  }

  // Specify dt if user has not specified it
  if (dt1 == 0) {
    dt = min_step / TIME_FUNC_OVERSAMPLING;
  } else {
    dt = dt1;
  }

  if (dt < 10 * FLOOR_THRESHOLD) TICK_ERROR("dt is too small, we currently cannot reach this precision");

  t0 = T[0];
  last_value_before_border = T[size - 1];
  switch (border_type) {
    case (BorderType::Border0):border_value = 0;
      break;
    case (BorderType::BorderConstant):border_value = border_value1;
      break;
    case (BorderType::BorderContinue):border_value = Y[size - 1];
      break;
  }

  // two cases
  // either dt divides length (for example T = [1, 2, 3] and dt = 0.5)
  // either dt don't (common case)
  const double length = T[size - 1] - T[0];

  if (length < dt) TICK_ERROR("``dt`` cannot be bigger than the time space you are considering");

  // We add 2 in order to be sure that we will have at least one value after the last given t
  // This value will be used to interpolate until the last given t
  ulong sample_size = (ulong) FLOOR(length / dt) + 2;

  sampled_y = SArrayDouble::new_ptr(sample_size);

  // We iterate over all given points and evaluate the function between them to fill sampled_y by dt step
  double t = t0;

  // Index of the left point selected
  double index_left = 0;
  // initialize with the first two points
  double t_left = T[index_left];
  double y_left = Y[index_left];
  double t_right = T[index_left + 1];
  double y_right = Y[index_left + 1];

  double *const sampled_y_ptr = (*sampled_y).data();
  for (ulong i = 0; i < sampled_y->size(); ++i) {
    if (t > t_right + FLOOR_THRESHOLD) {
      // Ensure we are not behind the last point. This might happen if dt does not divides length
      // In this case we keep the last two points to interpolate
      if (index_left < size - 2) {
        index_left += 1;
        t_left = T[index_left];
        y_left = Y[index_left];
        t_right = T[index_left + 1];
        y_right = Y[index_left + 1];
      }
    }

    sampled_y_ptr[i] = interpolation(t_left, y_left, t_right, y_right, t);
    t += dt;
  }

  support_right = sampled_y->size() * dt + t0;
}

TimeFunction::~TimeFunction() {
  if (sampled_y) sampled_y.reset();
  if (future_max) future_max.reset();
}

void TimeFunction::compute_future_max() {
  // Check this TimeFunction is not constant
  if (last_value_before_border < 0) return;

  ulong sample_size = sampled_y->size();
  future_max = SArrayDouble::new_ptr(sample_size);

  double previous_max = border_value;
  // condition is : i + 1 > 0 as an ulong cannot be negative
  for (ulong i = sample_size - 1; i + 1 > 0; --i) {
    (*future_max)[i] = std::max((*sampled_y)[i], previous_max);
    previous_max = (*future_max)[i];
  }
//    cout << "future max" << endl;
//    future_max->print();
}

double TimeFunction::value(double t) {
  // First handle if we are out of the border

  // this test must be the first one otherwise we might have problem with constant TimeFunctions
  if (t > last_value_before_border + FLOOR_THRESHOLD)
    return border_value;

    // TODO : which behavior do we want if t < t0
  else if (t < t0)
    return 0.0;
  else if (t < 0)
    return 0.0;

  const ulong i_left = get_index_(t);

  const double t_left = get_t_from_index_(i_left);
  const double y_left = (*sampled_y)[i_left];
  const double t_right = get_t_from_index_(i_left + 1);
  const double y_right = (*sampled_y)[i_left + 1];

  return interpolation(t_left, y_left, t_right, y_right, t);
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

  // First handle if we are out of the border
  if (t > last_value_before_border) return border_value;
  else if (t < t0) return (*future_max)[0];
  else if (t < 0) return (*future_max)[0];

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
    case (InterLinear):
      if (std::abs(t_left - t_right) < FLOOR_THRESHOLD) {
        return std::abs(y_left - y_right);
      } else {
        const double slope = (y_right - y_left) / (t_right - t_left);
        return std::abs(slope) * dt;
      }
    case (InterConstLeft):return FLOOR_THRESHOLD;
    case (InterConstRight):return FLOOR_THRESHOLD;
    default:return 0;
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
  for (ulong i = 0; i < sampled_y->size() - 1; ++i) {
    double y_left = (*sampled_y)[i];
    double y_right = (*sampled_y)[i + 1];

    // if we cross the last value we must handle this case
    double t_right = dt * (i + 1);
    double span;
    if (t_right < last_value_before_border + FLOOR_THRESHOLD) {
      span = dt;
    } else {
      span = std::max(0.0, last_value_before_border - dt * i);
      y_right = value(last_value_before_border);
    }

    switch (inter_mode) {
      case (InterLinear): {
        // The norm can be decomposed between a rectangle and a triangle
        const double rectangle_norm = y_left * span;
        const double triangle_norm = (y_right - y_left) * span / 2;
        norm += rectangle_norm + triangle_norm;
        break;
      }
      case (InterConstLeft): {
        // only right point matters
        norm += y_right * span;
        break;
      }
      case (InterConstRight): {
        // only left point matters
        norm += y_left * span;
        break;
      }
    }
  }
  return norm;
}

double TimeFunction::constant_left_interpolation(double t_left, double y_left,
                                                 double t_right, double y_right,
                                                 double t_value) {
  if (std::abs(t_value - t_left) < FLOOR_THRESHOLD)
    return y_left;
  else
    return y_right;
}

double TimeFunction::constant_right_interpolation(double t_left, double y_left,
                                                  double t_right, double y_right,
                                                  double t_value) {
  if (std::abs(t_value - t_right) < FLOOR_THRESHOLD)
    return y_right;
  else
    return y_left;
}

double TimeFunction::linear_interpolation(double t_left, double y_left,
                                          double t_right, double y_right,
                                          double t_value) {
  if (std::abs(t_left - t_right) < FLOOR_THRESHOLD) {
    return (y_left + y_right) / 2.0;
  } else {
    double slope = (y_right - y_left) / (t_right - t_left);
    return y_left + slope * (t_value - t_left);
  }
}

double TimeFunction::interpolation(double t_left, double y_left, double t_right, double y_right, double t) {
  // Second do the right interpolation
  switch (inter_mode) {
    case (InterLinear):return linear_interpolation(t_left, y_left, t_right, y_right, t);
    case (InterConstLeft):return constant_left_interpolation(t_left, y_left, t_right, y_right, t);
    case (InterConstRight):return constant_right_interpolation(t_left, y_left, t_right, y_right, t);
    default:throw std::runtime_error("Undefined interpolation mode");
  }
}

ulong TimeFunction::get_index_(double t) {
  return (ulong) FLOOR((t - t0) / dt);
}

double TimeFunction::get_t_from_index_(ulong i) {
  return t0 + dt * i;
}
