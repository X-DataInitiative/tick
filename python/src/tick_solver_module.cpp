#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common/tick_pybind11_arrays.h"
#include "tick/linear_model/model_hinge.h"
#include "tick/linear_model/model_linreg.h"
#include "tick/linear_model/model_logreg.h"
#include "tick/linear_model/model_poisreg.h"
#include "tick/linear_model/model_quadratic_hinge.h"
#include "tick/linear_model/model_smoothed_hinge.h"
#include "tick/prox/prox_binarsity.h"
#include "tick/prox/prox_elasticnet.h"
#include "tick/prox/prox_equality.h"
#include "tick/prox/prox_group_l1.h"
#include "tick/prox/prox_l1.h"
#include "tick/prox/prox_l1w.h"
#include "tick/prox/prox_l2.h"
#include "tick/prox/prox_l2sq.h"
#include "tick/prox/prox_multi.h"
#include "tick/prox/prox_positive.h"
#include "tick/prox/prox_separable.h"
#include "tick/prox/prox_slope.h"
#include "tick/prox/prox_sorted_l1.h"
#include "tick/prox/prox_tv.h"
#include "tick/solver/asaga.h"
#include "tick/solver/adagrad.h"
#include "tick/solver/enums.h"
#include "tick/solver/saga.h"
#include "tick/solver/sgd.h"
#include "tick/solver/sdca.h"
#include "tick/solver/svrg.h"
#include "tick/robust/model_absolute_regression.h"
#include "tick/robust/model_epsilon_insensitive.h"
#include "tick/robust/model_generalized_linear_with_intercepts.h"
#include "tick/robust/model_huber.h"
#include "tick/robust/model_linreg_with_intercepts.h"
#include "tick/robust/model_modified_huber.h"
#include "tick/survival/model_coxreg_partial_lik.h"
#include "tick/survival/model_sccs.h"

namespace py = pybind11;

namespace {

template <typename SolverType, typename Scalar>
void multi_solve_svrg_impl(const py::iterable &solver_objects,
                           size_t epochs, py::object threads_object,
                           py::object starters_object) {
  std::vector<std::shared_ptr<SolverType>> solver_holders;
  std::vector<SolverType *> solvers;
  solver_holders.reserve(static_cast<size_t>(py::len(solver_objects)));
  solvers.reserve(static_cast<size_t>(py::len(solver_objects)));

  for (const py::handle &item : solver_objects) {
    auto solver = item.cast<std::shared_ptr<SolverType>>();
    solver_holders.push_back(solver);
    solvers.push_back(solver.get());
  }

  if (threads_object.is_none()) {
    MultiSVRG<Scalar, Scalar>::multi_solve(solvers, epochs);
    return;
  }

  const auto threads = threads_object.cast<size_t>();
  if (starters_object.is_none()) {
    MultiSVRG<Scalar, Scalar>::multi_solve(solvers, epochs, threads);
    return;
  }

  auto starters =
      starters_object.cast<std::vector<std::shared_ptr<SArray<Scalar>>>>();
  MultiSVRG<Scalar, Scalar>::multi_solve(solvers, starters, epochs, threads);
}

template <typename SolverType, typename Scalar>
void bind_sto_solver_base(py::module_ &m, const char *name) {
  using ModelType = TModel<Scalar, Scalar>;
  using ProxType = TProx<Scalar, Scalar>;

  auto cls = py::class_<SolverType, std::shared_ptr<SolverType>>(m, name)
      .def("set_model", &SolverType::set_model, py::arg("model"),
           py::return_value_policy::reference_internal)
      .def("set_prox", &SolverType::set_prox, py::arg("prox"),
           py::return_value_policy::reference_internal)
      .def("reset", &SolverType::reset)
      .def("solve", &SolverType::solve, py::arg("n_epochs") = 1)
      .def("get_minimizer",
           [](SolverType &self, Array<Scalar> &out) { self.get_minimizer(out); },
           py::arg("out"))
      .def("get_iterate",
           [](SolverType &self, Array<Scalar> &out) { self.get_iterate(out); },
           py::arg("out"))
      .def("set_starting_iterate",
           py::overload_cast<Array<Scalar> &>(&SolverType::set_starting_iterate),
           py::arg("iterate"))
      .def("get_tol", &SolverType::get_tol)
      .def("set_tol", &SolverType::set_tol, py::arg("tol"))
      .def("get_epoch_size", &SolverType::get_epoch_size)
      .def("set_epoch_size", &SolverType::set_epoch_size, py::arg("epoch_size"))
      .def("get_t", &SolverType::get_t)
      .def("get_rand_type", &SolverType::get_rand_type)
      .def("set_rand_type", &SolverType::set_rand_type, py::arg("rand_type"))
      .def("get_rand_max", &SolverType::get_rand_max)
      .def("set_rand_max", &SolverType::set_rand_max, py::arg("rand_max"))
      .def("set_seed", &SolverType::set_seed, py::arg("seed"))
      .def("get_record_every", &SolverType::get_record_every)
      .def("set_record_every", &SolverType::set_record_every,
           py::arg("record_every"))
      .def("get_time_history", &SolverType::get_time_history)
      .def("get_objectives", &SolverType::get_objectives)
      .def("get_objective", &SolverType::get_objective)
      .def("set_prev_obj", &SolverType::set_prev_obj, py::arg("obj"),
           py::return_value_policy::reference_internal)
      .def("set_first_obj", &SolverType::set_first_obj, py::arg("obj"),
           py::return_value_policy::reference_internal)
      .def("get_first_obj", &SolverType::get_first_obj)
      .def("get_epoch_history", &SolverType::get_epoch_history)
      .def("get_iterate_history", &SolverType::get_iterate_history)
      .def("get_model", &SolverType::get_model)
      .def("get_prox", &SolverType::get_prox);
  tick::pybind::enable_cereal_pickle<SolverType>(cls);
}

template <typename SolverType, typename BaseType, typename Scalar>
void bind_sgd_like(py::module_ &m, const char *name) {
  auto cls = py::class_<SolverType, std::shared_ptr<SolverType>, BaseType>(m,
                                                                           name)
      .def(py::init<ulong, Scalar, RandType, Scalar, int, int>(),
           py::arg("epoch_size"), py::arg("tol"), py::arg("rand_type"),
           py::arg("step"), py::arg("record_every") = 1,
           py::arg("seed") = -1)
      .def("get_step", &SolverType::get_step)
      .def("set_step", &SolverType::set_step, py::arg("step"))
      .def("compare",
           [](SolverType &self, SolverType &other) {
             return static_cast<bool>(self.compare(other));
           });
  tick::pybind::enable_cereal_pickle<SolverType>(cls);
}

template <typename SolverType, typename BaseType, typename Scalar>
void bind_svrg_like(py::module_ &m, const char *name) {
  auto cls = py::class_<SolverType, std::shared_ptr<SolverType>, BaseType>(m,
                                                                           name)
      .def(py::init<size_t, Scalar, RandType, Scalar, size_t, int, size_t,
                    SVRG_VarianceReductionMethod, SVRG_StepType>(),
           py::arg("epoch_size"), py::arg("tol"), py::arg("rand_type"),
           py::arg("step"), py::arg("record_every") = 1,
           py::arg("seed") = -1, py::arg("n_threads") = 1,
           py::arg("variance_reduction") =
               SVRG_VarianceReductionMethod::Last,
           py::arg("step_method") = SVRG_StepType::Fixed)
      .def("get_step", &SolverType::get_step)
      .def("set_step", &SolverType::set_step, py::arg("step"))
      .def("get_variance_reduction", &SolverType::get_variance_reduction)
      .def("set_variance_reduction", &SolverType::set_variance_reduction,
           py::arg("variance_reduction"))
      .def("get_step_type", &SolverType::get_step_type)
      .def("set_step_type", &SolverType::set_step_type, py::arg("step_type"))
      .def("compare",
           [](SolverType &self, SolverType &other) {
             return static_cast<bool>(self.compare(other));
           });
  tick::pybind::enable_cereal_pickle<SolverType>(cls);
}

template <typename SolverType, typename BaseType, typename Scalar>
void bind_sdca_like(py::module_ &m, const char *name) {
  auto cls = py::class_<SolverType, std::shared_ptr<SolverType>, BaseType>(m,
                                                                           name)
      .def(py::init<Scalar, ulong, Scalar, RandType, int, int>(),
           py::arg("l_l2sq"), py::arg("epoch_size") = 0, py::arg("tol") = 0,
           py::arg("rand_type") = RandType::unif,
           py::arg("record_every") = 1, py::arg("seed") = -1)
      .def("get_l_l2sq", &SolverType::get_l_l2sq)
      .def("set_l_l2sq", &SolverType::set_l_l2sq, py::arg("l_l2sq"))
      .def("get_dual_vector", &SolverType::get_dual_vector)
      .def("get_primal_vector", &SolverType::get_primal_vector)
      .def("compare",
           [](SolverType &self, SolverType &other) {
             return static_cast<bool>(self.compare(other));
           });
  tick::pybind::enable_cereal_pickle<SolverType>(cls);
}

template <typename SolverType, typename BaseType, typename Scalar>
void bind_adagrad_like(py::module_ &m, const char *name) {
  auto cls = py::class_<SolverType, std::shared_ptr<SolverType>, BaseType>(m,
                                                                           name)
      .def(py::init<ulong, Scalar, RandType, Scalar, int, int>(),
           py::arg("epoch_size"), py::arg("tol"), py::arg("rand_type"),
           py::arg("step"), py::arg("record_every") = 1,
           py::arg("seed") = -1)
      .def("compare",
           [](SolverType &self, SolverType &other) {
             return static_cast<bool>(self.compare(other));
           });
  tick::pybind::enable_cereal_pickle<SolverType>(cls);
}

template <typename SolverType, typename BaseType, typename Scalar>
void bind_saga_like(py::module_ &m, const char *name) {
  auto cls = py::class_<SolverType, std::shared_ptr<SolverType>, BaseType>(m,
                                                                           name)
      .def(py::init<ulong, Scalar, RandType, Scalar, int, int>(),
           py::arg("epoch_size"), py::arg("tol"), py::arg("rand_type"),
           py::arg("step"), py::arg("record_every") = 1,
           py::arg("seed") = -1)
      .def("get_step", &SolverType::get_step)
      .def("set_step", &SolverType::set_step, py::arg("step"))
      .def("compare",
           [](SolverType &self, SolverType &other) {
             return static_cast<bool>(self.compare(other));
           });
  tick::pybind::enable_cereal_pickle<SolverType>(cls);
}

template <typename SolverType, typename BaseType, typename Scalar>
void bind_atomic_saga_like(py::module_ &m, const char *name) {
  auto cls = py::class_<SolverType, std::shared_ptr<SolverType>, BaseType>(m,
                                                                           name)
      .def(py::init([](ulong epoch_size, Scalar tol, RandType rand_type,
                       Scalar step, int record_every, int seed,
                       int n_threads) {
             return std::make_shared<SolverType>(epoch_size, tol, rand_type,
                                                 step, record_every, seed,
                                                 n_threads);
           }),
           py::arg("epoch_size"), py::arg("tol"), py::arg("rand_type"),
           py::arg("step"), py::arg("record_every") = 1,
           py::arg("seed") = -1, py::arg("n_threads") = 2)
      .def("get_step", &SolverType::get_step)
      .def("set_step", &SolverType::set_step, py::arg("step"))
      .def("compare",
           [](SolverType &self, SolverType &other) {
             return static_cast<bool>(self.compare(other));
           });
  tick::pybind::enable_cereal_pickle<SolverType>(cls);
}

}  // namespace

PYBIND11_MODULE(solver, m) {
  tick::pybind::ensure_numpy_imported();

  py::enum_<RandType>(m, "RandType")
      .value("unif", RandType::unif)
      .value("perm", RandType::perm);
  m.attr("RandType_unif") = py::cast(RandType::unif);
  m.attr("RandType_perm") = py::cast(RandType::perm);

  py::enum_<SVRG_VarianceReductionMethod>(m, "SVRG_VarianceReductionMethod")
      .value("Last", SVRG_VarianceReductionMethod::Last)
      .value("Average", SVRG_VarianceReductionMethod::Average)
      .value("Random", SVRG_VarianceReductionMethod::Random);
  m.attr("SVRG_VarianceReductionMethod_Last") =
      py::cast(SVRG_VarianceReductionMethod::Last);
  m.attr("SVRG_VarianceReductionMethod_Average") =
      py::cast(SVRG_VarianceReductionMethod::Average);
  m.attr("SVRG_VarianceReductionMethod_Random") =
      py::cast(SVRG_VarianceReductionMethod::Random);

  py::enum_<SVRG_StepType>(m, "SVRG_StepType")
      .value("Fixed", SVRG_StepType::Fixed)
      .value("BarzilaiBorwein", SVRG_StepType::BarzilaiBorwein);
  m.attr("SVRG_StepType_Fixed") = py::cast(SVRG_StepType::Fixed);
  m.attr("SVRG_StepType_BarzilaiBorwein") =
      py::cast(SVRG_StepType::BarzilaiBorwein);

  bind_sto_solver_base<TStoSolver<double, double>, double>(m, "StoSolverDouble");
  bind_sto_solver_base<TStoSolver<float, float>, float>(m, "StoSolverFloat");

  bind_sgd_like<TSGD<double, double>, TStoSolver<double, double>, double>(
      m, "SGDDouble");
  bind_sgd_like<TSGD<float, float>, TStoSolver<float, float>, float>(
      m, "SGDFloat");

  bind_adagrad_like<TAdaGrad<double>, TStoSolver<double, double>, double>(
      m, "AdaGradDouble");
  bind_adagrad_like<TAdaGrad<float>, TStoSolver<float, float>, float>(
      m, "AdaGradFloat");

  bind_svrg_like<TSVRG<double, double>, TStoSolver<double, double>, double>(
      m, "SVRGDouble");
  bind_svrg_like<TSVRG<float, float>, TStoSolver<float, float>, float>(
      m, "SVRGFloat");

  bind_sdca_like<TSDCA<double, double>, TStoSolver<double, double>, double>(
      m, "SDCADouble");
  bind_sdca_like<TSDCA<float, float>, TStoSolver<float, float>, float>(
      m, "SDCAFloat");

  bind_saga_like<TSAGA<double>, TStoSolver<double, double>, double>(
      m, "SAGADouble");
  bind_saga_like<TSAGA<float>, TStoSolver<float, float>, float>(m,
                                                                 "SAGAFloat");
  bind_atomic_saga_like<AtomicSAGADouble, TStoSolver<double, double>, double>(
      m, "AtomicSAGADouble");
  bind_atomic_saga_like<AtomicSAGAFloat, TStoSolver<float, float>, float>(
      m, "AtomicSAGAFloat");

  m.def(
      "multi_solve_svrg_double",
      [](const py::iterable &solver_objects, size_t epochs,
         py::object threads_object, py::object starters_object) {
        multi_solve_svrg_impl<SVRGDouble, double>(solver_objects, epochs,
                                                  threads_object,
                                                  starters_object);
      },
      py::arg("solver_objects"), py::arg("epochs"),
      py::arg("threads") = py::none(), py::arg("starters") = py::none());
  m.def(
      "multi_solve_svrg_float",
      [](const py::iterable &solver_objects, size_t epochs,
         py::object threads_object, py::object starters_object) {
        multi_solve_svrg_impl<SVRGFloat, float>(solver_objects, epochs,
                                                threads_object,
                                                starters_object);
      },
      py::arg("solver_objects"), py::arg("epochs"),
      py::arg("threads") = py::none(), py::arg("starters") = py::none());
}
