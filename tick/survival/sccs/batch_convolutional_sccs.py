
from tick.survival.convolutional_sccs import *



class BatchConvSCCS(ConvSCCS):


    """BatchConvSCCS provides parallel batch solving for ConvSCCS k_fold
        and bootstrap

    Parameters
    ----------
    batch_size : integer
     for kfold must be multiple of n_fold, and divisor of n_cv_iter
     for bootstrap must be <= rep
    """

    def __init__(self, n_lags: np.array, penalized_features: np.array = None,
                 C_tv=None, C_group_l1=None, step: float = None,
                 tol: float = 1e-5, max_iter: int = 100, verbose: bool = False,
                 print_every: int = 10, record_every: int = 10,
                 random_state: int = None, batch_size = 1):
        _, _, _, kvs = inspect.getargvalues(inspect.currentframe())
        object.__setattr__(self, "batch_size", batch_size)
        del kvs['batch_size']
        ConvSCCS.__init__(**kvs)

    def _multi_fit(self, model_list, coeffs_list, C_s, n_folds):
        solvers, proxes = ([] for i in range(2)) # 2 on the left
        for c in range(len(C_s)):
            self.C_tv, self.C_group_l1 = C_s[c]
            for i in range(n_folds):
                index = i + (c * n_folds)
                proxes.append(self._construct_prox_obj())
                solvers.append(self._construct_solver_obj_with_class(*self._solver_info))
                solvers[-1].step = self.step
                solvers[-1].set_model(model_list[index]).set_prox(proxes[-1])
                solvers[-1]._solver.set_epoch_size(model_list[index]._model.get_epoch_size())
        # no point calling "set_start_iterate" as it is always 0s
        return solvers[0].multi_solve(coeffs_list, solvers, self.max_iter, self.batch_size, set_start=False)

    def fit_kfold_cv(self, features, labels, censoring, C_tv_range: tuple = (),
                         C_group_l1_range: tuple = (), logscale=True,
                         n_cv_iter: int = 30, n_folds: int = 3,
                         shuffle: bool = True, confidence_intervals: bool = False,
                         n_samples_bootstrap: int = 100, confidence_level: float = .95):
        p_features, p_labels, p_censoring = self._prefit(
            features, labels, censoring)
        # split the data with stratified KFold
        kf = StratifiedKFold(n_folds, shuffle, self.random_state)
        labels_interval = np.nonzero(p_labels)[1]

        # Training loop
        model_global_parameters = {
            "n_intervals": self.n_intervals,
            "n_lags": self.n_lags,
            "n_features": self.n_features,
        }
        cv_tracker = CrossValidationTracker(model_global_parameters)
        C_tv_generator, C_group_l1_generator = self._construct_generator_obj(
            C_tv_range, C_group_l1_range, logscale)

        self._set("_fitted", True)
        if n_folds > self.batch_size:
            self.batch_size = n_folds
        batch_size = self.batch_size
        if batch_size > (n_cv_iter * n_folds):
            batch_size = (n_cv_iter * n_folds)
        batches = (n_cv_iter // batch_size) * n_folds
        n_fold_groups = batch_size // n_folds
        for i in range(batches):
            C_s, score_vals, indexes = ([] for i in range(3)) # 3 on the left
            model_list, coeffs_list  = ([] for i in range(2)) # 2 on the left
            for i in range(batch_size):
                model_list.append(self._construct_model_obj())
                coeffs_list.append(np.zeros(self.n_coeffs))
            for group in range(n_fold_groups):
                C_s.append((C_tv_generator.rvs(1)[0], C_group_l1_generator.rvs(1)[0]))
                indexes.append(list(kf.split(p_features, labels_interval)))
                for n_job in range(n_folds):
                    index = n_job + (group * n_folds)
                    train_index, test_index = indexes[-1][n_job]
                    train = itemgetter(*train_index.tolist())
                    test = itemgetter(*test_index.tolist())
                    X_train, X_test = list(train(p_features)), list(test(p_features))
                    y_train, y_test = list(train(p_labels)), list(test(p_labels))
                    censoring_train, censoring_test = p_censoring[train_index], \
                        p_censoring[test_index]
                    model_list[index].fit(X_train, y_train, censoring_train)
                    score_vals.append((X_test, y_test, censoring_test))

            minimizers = self._multi_fit(model_list, coeffs_list, C_s, n_folds)
            for group in range(n_fold_groups):
                train_scores, test_scores = ([] for i in range(2)) # 2 on the left
                for n_job in range(n_folds):
                    index = n_job + (group * n_folds)
                    train_scores.append(model_list[index].loss(minimizers[index]))
                    X_test, y_test, censoring_test = score_vals[index]
                    test_scores.append(self._construct_model_obj().fit(
                            X_test, y_test, censoring_test).loss(minimizers[index]))
                self.C_tv, self.C_group_l1 = C_s[group]
                cv_tracker.log_cv_iteration(self.C_tv, self.C_group_l1,
                                            np.array(train_scores),
                                            np.array(test_scores))

        # refit best model on all the data
        best_parameters = cv_tracker.find_best_params()
        self.C_tv = best_parameters["C_tv"]
        self.C_group_l1 = best_parameters["C_group_l1"]
        self._set('_coeffs', np.zeros(self.n_coeffs))
        self._model_obj.fit(p_features, p_labels, p_censoring)
        coeffs, bootstrap_ci = self._postfit(
            p_features, p_labels, p_censoring, True, confidence_intervals,
            n_samples_bootstrap, confidence_level)
        cv_tracker.log_best_model(self.C_tv, self.C_group_l1,
                                  self._coeffs.tolist(), self.score(),
                                  self.confidence_intervals)
        return self.coeffs, cv_tracker

    def _bootstrap_multi_fit(self, model_list, coeffs_list):
        solvers, proxes  = ([] for i in range(2)) # 2 on the left
        for i in range(len(coeffs_list)):
            proxes.append(self._construct_prox_obj(coeffs=coeffs_list[i], project=True))
            solvers.append(self._construct_solver_obj_with_class(*self._solver_info))
            solvers[-1].step = self.step
            solvers[-1].set_model(model_list[i]).set_prox(proxes[-1])
            solvers[-1]._solver.set_epoch_size(model_list[i]._model.get_epoch_size())

        return solvers[0].multi_solve(coeffs_list, solvers, self.max_iter, self.batch_size, True)

    def _bootstrap(self, p_features, p_labels, p_censoring, coeffs, rep, confidence):
        # WARNING: _bootstrap inputs are already preprocessed p_features,
        # p_labels and p_censoring
        # Coeffs here are assumed to be an array (same object than self._coeffs)
        if confidence <= 0 or confidence >= 1:
            raise ValueError("`confidence_level` should be in (0, 1)")
        confidence = 1 - confidence
        if not self._fitted:
            raise RuntimeError('You must fit the model first')

        model_list, bootstrap_coeffs = ([] for i in range(2)) # 2 on the left
        for i in range(self.batch_size):
            model_list.append(self._construct_model_obj())

        sim = SimuSCCS(self.n_cases, self.n_intervals, self.n_features,
                       self.n_lags, coeffs=self._format_coeffs(coeffs))

        def n_bootstrap(jobs):
            coeffs_list = []
            for k in range(jobs):
                y = sim._simulate_multinomial_outcomes(p_features, coeffs)
                model_list[k].fit(p_features, y, p_censoring)
                coeffs_list.append(self._coeffs.copy())
            solved_coeffs = self._bootstrap_multi_fit(model_list, coeffs_list)
            for k in range(jobs):
                bootstrap_coeffs.append(solved_coeffs[k])

        for i in range(rep // self.batch_size):
            n_bootstrap(self.batch_size)
        modulo = rep % self.batch_size
        if modulo > 0:
            n_bootstrap(modulo)

        bootstrap_coeffs = np.exp(np.array(bootstrap_coeffs))
        bootstrap_coeffs.sort(axis=0)
        lower_bound = np.log(bootstrap_coeffs[int(
            np.floor(rep * confidence / 2))])
        upper_bound = np.log(bootstrap_coeffs[int(
            np.floor(rep * (1 - confidence / 2)))])
        return Confidence_intervals(
            self._format_coeffs(coeffs), self._format_coeffs(lower_bound),
            self._format_coeffs(upper_bound), confidence)


