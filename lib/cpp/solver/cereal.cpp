
// License: BSD 3 clause

/*
*  Windows DLLs have a strange effect with static objects
*   so that at least for the case of cereal there is 
*   numerous instances of the same static object
*   which causes all types to need to be registed per DLL
*/

#ifdef _WIN32

#include "tick/prox/prox_binarsity.h"
#include "tick/prox/prox_elasticnet.h"
#include "tick/prox/prox_equality.h"
#include "tick/prox/prox_group_l1.h"
#include "tick/prox/prox_l1.h"
#include "tick/prox/prox_l1w.h"
#include "tick/prox/prox_l2.h"
#include "tick/prox/prox_l2sq.h"
#include "tick/prox/prox_positive.h"
#include "tick/prox/prox_slope.h"
#include "tick/prox/prox_sorted_l1.h"
#include "tick/prox/prox_tv.h"
#include "tick/prox/prox_zero.h"

#include "tick/linear_model/model_hinge.h"
#include "tick/linear_model/model_linreg.h"
#include "tick/linear_model/model_logreg.h"
#include "tick/linear_model/model_poisreg.h"
#include "tick/linear_model/model_quadratic_hinge.h"
#include "tick/linear_model/model_smoothed_hinge.h"

#include "tick/robust/model_absolute_regression.h"
#include "tick/robust/model_epsilon_insensitive.h"
#include "tick/robust/model_generalized_linear_with_intercepts.h"
#include "tick/robust/model_huber.h"
#include "tick/robust/model_linreg_with_intercepts.h"
#include "tick/robust/model_modified_huber.h"

#endif
