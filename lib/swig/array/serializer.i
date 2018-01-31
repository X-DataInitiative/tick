// License: BSD 3 clause


%include <std_string.i>
%include <std_shared_ptr.i>

%{
#include "tick/array/serializer.h"
%}


void tick_float_array_to_file(std::string _file, const ArrayFloat& array);
SArrayFloatPtr tick_float_array_from_file(std::string _file);

void tick_float_array2d_to_file(std::string _file, const ArrayFloat2d& array);
SArrayFloat2dPtr tick_float_array2d_from_file(std::string _file);

void tick_float_sparse2d_to_file(std::string _file, const SparseArrayFloat2d& array);
SSparseArrayFloat2dPtr tick_float_sparse2d_from_file(std::string _file);


void tick_double_array_to_file(std::string _file, const ArrayDouble& array);
SArrayDoublePtr tick_double_array_from_file(std::string _file);

void tick_double_array2d_to_file(std::string _file, const ArrayDouble2d& array);
SArrayDouble2dPtr tick_double_array2d_from_file(std::string _file);

void tick_double_sparse2d_to_file(std::string _file, const SparseArrayDouble2d& array);
SSparseArrayDouble2dPtr tick_double_sparse2d_from_file(std::string _file);

