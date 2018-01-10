#ifndef LIB_INCLUDE_TICK_BASE_TIME_FUNC_H_
#define LIB_INCLUDE_TICK_BASE_TIME_FUNC_H_

// License: BSD 3 clause

/** @file */

#include "defs.h"

#include <cmath>
#include <memory>

#include "tick/array/array.h"
#include "tick/array/sarray.h"

#include <cereal/cereal.hpp>

// TODO: Do an abstract class... then small classes for constant TF, interpolated, dichotomic...

//! @brief this class allows us to extrapolate points samples into continuous temporal function
class DLL_PUBLIC TimeFunction {
 public:
    //! @brief The different interpolation mode
    enum class InterMode {
        InterLinear = 0,
        InterConstRight,
        InterConstLeft
    };

    //! @brief The different border effects
    enum class BorderType {
        Border0 = 0,
        BorderConstant,
        BorderContinue,
        Cyclic,
    };

 public:
    static const InterMode DEFAULT_INTER = InterMode::InterLinear;
    static const BorderType DEFAULT_BORDER = BorderType::Border0;

 public:
    TimeFunction(const ArrayDouble& Y, BorderType type, InterMode mode, double dt, double border_value);

    // Does nothing but allowing user to specify his arguments in another order
    TimeFunction(const ArrayDouble &T, const ArrayDouble &Y, double dt);

    // Main constructor
    TimeFunction(const ArrayDouble &T, const ArrayDouble &Y,
                 BorderType type = DEFAULT_BORDER, InterMode mode = DEFAULT_INTER,
                 double dt = 0.0, double border_value = 0.0);

    // Constant constuctor
    explicit TimeFunction(double y = 0.0);

    // No virtual, otherwise it generates a warning while using in std::vector
    ~TimeFunction();

    // Copy Constructor
    TimeFunction(const TimeFunction &) = default;

    // TODO: n_points constructor

 private :
    InterMode inter_mode;
    BorderType border_type;

 public:
    InterMode get_inter_mode() const { return inter_mode; }

    BorderType get_border_type() const { return border_type; }

    SArrayDoublePtr get_sampled_y() const { return sampled_y; }

    SArrayDoublePtr get_future_max() const { return future_max; }

    double get_dt() const { return dt; }

    double get_support_right() const { return support_right; }

    // interpolation function
    double interpolation(double x_left, double y_left, double x_right, double y_right, double x_value);

    // Call function
    double value(double t);

    SArrayDoublePtr value(ArrayDouble &array);

    double future_bound(double t);

    void compute_future_max();

    SArrayDoublePtr future_bound(ArrayDouble &array);

    double max_error(double t);

    double get_norm();

 private:
    SArrayDoublePtr sampled_y;
    SArrayDoublePtr future_max;
    double t0;
    double dt;
    double support_right;
    double last_value_before_border;
    double border_value;

    inline ulong get_index_(double t);

    inline double get_t_from_index_(ulong i);

    inline double constant_left_interpolation(double x_left, double y_left, double x_right, double y_right,
                                              double x_value);

    inline double constant_right_interpolation(double x_left, double y_left, double x_right, double y_right,
                                               double x_value);

    inline double linear_interpolation(double x_left, double y_left, double x_right, double y_right, double x_value);

 public:
    double get_border_value() { return border_value; }

  template<class Archive>
  void load(Archive & ar) {
    ArrayDouble temp_sampled_y;
    ArrayDouble temp_future_max;
    ar(cereal::make_nvp("sampled_y", temp_sampled_y));
    ar(cereal::make_nvp("future_max", temp_future_max));

    sampled_y = temp_sampled_y.as_sarray_ptr();

    // If future_max is empty, we let it be a nullptr instead of initializing a new array
    future_max = temp_future_max.size() == 0 ? nullptr : temp_future_max.as_sarray_ptr();

    ar(CEREAL_NVP(inter_mode));
    ar(CEREAL_NVP(border_type));
    ar(CEREAL_NVP(t0));
    ar(CEREAL_NVP(dt));
    ar(CEREAL_NVP(support_right));
    ar(CEREAL_NVP(last_value_before_border));
    ar(CEREAL_NVP(border_value));
  }

  template<class Archive>
  void save(Archive & ar) const {
    ar(cereal::make_nvp("sampled_y", sampled_y.get() ? *sampled_y : ArrayDouble(0)));
    ar(cereal::make_nvp("future_max", future_max.get() ? *future_max : ArrayDouble(0)));

    ar(CEREAL_NVP(inter_mode));
    ar(CEREAL_NVP(border_type));
    ar(CEREAL_NVP(t0));
    ar(CEREAL_NVP(dt));
    ar(CEREAL_NVP(support_right));
    ar(CEREAL_NVP(last_value_before_border));
    ar(CEREAL_NVP(border_value));
  }
};

typedef std::shared_ptr<TimeFunction> TimeFunctionPtr;

#endif  // LIB_INCLUDE_TICK_BASE_TIME_FUNC_H_
