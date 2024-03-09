
%module tick_cpp

%include tick/array/array_module.i
%include tick/array_test/array_test_module.i

%include tick/base/base_module.i

%include tick/random/crandom_module.i

%include tick/base_model/base_model_module.i
%include tick/linear_model/linear_model_module.i

%include tick/prox/prox_module.i

%include tick/hawkes/model/hawkes_model_module.i
%include tick/hawkes/simulation/hawkes_simulation_module.i
%include tick/hawkes/inference/hawkes_inference_module.i
%include tick/preprocessing/preprocessing_module.i
%include tick/robust/robust_module.i
%include tick/solver/solver_module.i
%include tick/survival/survival_module.i
