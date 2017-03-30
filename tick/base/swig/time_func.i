%{
#include "time_func.h"
%}

class TimeFunction {
    public:
        // The different interpolation mode
        typedef enum {
            InterLinear = 0,
            InterConstRight,
            InterConstLeft
        } InterMode;

        // The different border effects
        typedef enum {
            Border0 = 0,
            BorderConstant,
            BorderContinue
        } BorderType;

    public:
        static const InterMode DEFAULT_INTER;
        static const BorderType DEFAULT_BORDER;

    public:
        // Main constructor
        TimeFunction(const ArrayDouble &T, const ArrayDouble &Y,
                     BorderType type, InterMode mode,
                     double dt, double border_value);

        // Constant constructor
        TimeFunction(double y = 0.0);

        // Call function
        double value(double t);
        SArrayDoublePtr value(ArrayDouble &array);

        double future_bound(double t);
        SArrayDoublePtr future_bound(ArrayDouble &array);

        double max_error(double t);

        void compute_future_max();

        double get_norm();
        TimeFunction::InterMode get_inter_mode();
        TimeFunction::BorderType get_border_type();
        double get_border_value();
        SArrayDoublePtr get_sampled_y();
        SArrayDoublePtr get_future_max();
        double get_dt();
};

TICK_MAKE_PICKLABLE(TimeFunction, 0.0);
