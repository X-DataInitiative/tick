import tick

import os
import pandas as pd
import numpy as np
from zipfile import ZipFile
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from experience.poisreg_sdca import ModelPoisRegSDCA
from tick.dataset.download_helper import download_tick_dataset, get_data_home
from tick.optim.model import ModelPoisReg
from tick.optim.prox import ProxZero, ProxL2Sq, ProxPositive
from tick.optim.solver import SDCA, Newton, LBFGSB, SVRG, SCPG
from tick.plot import plot_history, stems
from tick.preprocessing.features_binarizer import FeaturesBinarizer


def fetch_uci_dataset(dataset_path, data_filename, sep=',', header=0):
    base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/%s'
    cache_path = os.path.join(get_data_home(), dataset_path)

    if not os.path.exists(cache_path):
        cache_path = download_tick_dataset(dataset_path, base_url=base_url)

    try:
        df = pd.read_csv(cache_path, header=header, sep=sep,
                         low_memory=False)

    except ValueError:
        zip_file = ZipFile(cache_path)
        with zip_file.open(data_filename) as data_file:
            df = pd.read_csv(data_file, header=header, sep=sep,
                             low_memory=False)
    return df


def fetch_blog_dataset(n_samples=52397):
    dataset_path = '00304/BlogFeedback.zip'
    data_filename = "blogData_train.csv"  # "blogData_test-2012.03.25.00_00.csv"  #

    original_df = fetch_uci_dataset(dataset_path, data_filename, header=-1)
    shuffled_df = shuffle(original_df)

    data = shuffled_df.head(n_samples).values
    features = data[:, :-1]
    labels = data[:, -1]
    labels = np.ascontiguousarray(labels)

    mask = (features.max(axis=0) - features.min(axis=0)) != 0
    features = features[:, mask]
    features = (features - features.min(axis=0)) / \
               (features.max(axis=0) - features.min(axis=0))

    features = np.ascontiguousarray(features)

    return features, labels


def fetch_news_popularity_dataset(n_samples=39797):
    dataset_path = '00332/OnlineNewsPopularity.zip'
    data_filename = "OnlineNewsPopularity/OnlineNewsPopularity.csv"

    original_df = fetch_uci_dataset(dataset_path, data_filename, header=-1)
    shuffled_df = shuffle(original_df)

    data = shuffled_df.head(n_samples).values
    features = data[:, 1:-1].astype(float)
    labels = data[:, -1].astype(float)
    labels = np.ascontiguousarray(labels)

    mask = (features.max(axis=0) - features.min(axis=0)) != 0
    features = features[:, mask]
    features = (features - features.min(axis=0)) / \
               (features.max(axis=0) - features.min(axis=0))

    features = np.ascontiguousarray(features)

    return features, labels


def fetch_las_vegas_dataset(n_samples=504):
    dataset_path = '00397/LasVegasTripAdvisorReviews-Dataset.csv'
    data_filename = "LasVegasTripAdvisorReviews-Dataset.csv"

    original_df = fetch_uci_dataset(dataset_path, data_filename, sep=';')
    shuffled_df = shuffle(original_df)
    shuffled_df = shuffled_df.head(n_samples)

    labels = shuffled_df['Score'].values.astype(float)
    labels = np.ascontiguousarray(labels)

    features = shuffled_df.drop('Score', axis=1).values
    binarizer = FeaturesBinarizer(remove_first=True)
    features = binarizer.fit_transform(features)
    features = np.ascontiguousarray(features.toarray().astype(float))

    return features, labels


def fetch_wine_datase(n_samples=4998):
    dataset_path = 'wine-quality/winequality-white.csv'
    data_filename = "winequality-white.csv"

    original_df = fetch_uci_dataset(dataset_path, data_filename, sep=';')
    shuffled_df = shuffle(original_df)
    shuffled_df = shuffled_df.head(n_samples)

    labels = shuffled_df['quality'].values.astype(float)
    labels = np.ascontiguousarray(labels)

    features = shuffled_df.drop('quality', axis=1).values
    features = (features - features.min(axis=0)) / \
               (features.max(axis=0) - features.min(axis=0))
    features = np.ascontiguousarray(features.astype(float))

    return features, labels


def fetch_crime_dataset(n_samples=2215):
    dataset_path = '00211/CommViolPredUnnormalizedData.txt'
    data_filename = "CommViolPredUnnormalizedData.txt"

    original_df = fetch_uci_dataset(dataset_path, data_filename, header=-1)
    shuffled_df = shuffle(original_df)
    shuffled_df = shuffled_df.head(n_samples)

    labels = shuffled_df[129].values.astype(float)
    labels = np.ascontiguousarray(labels)

    features_columns = list(shuffled_df.columns[5:129])

    for col in shuffled_df[features_columns]:
        if shuffled_df[col].dtype == 'object':
            n_missing_values = sum(shuffled_df[col] == '?')
            prop_missing_values = n_missing_values / len(shuffled_df[col])
            if prop_missing_values > 0.5:
                features_columns.remove(col)
            else:
                # We put the number of the line before
                missing_indices = (shuffled_df[col] == '?').values
                previous_indices = np.hstack((missing_indices[-1],
                                              missing_indices[:-1]))
                shuffled_df[col][missing_indices] = \
                    float(shuffled_df[col][previous_indices])
                shuffled_df[col] = shuffled_df[col].astype(float)

    features = shuffled_df[features_columns].values
    features = (features - features.min(axis=0)) / \
               (features.max(axis=0) - features.min(axis=0))
    features = np.ascontiguousarray(features.astype(float))

    return features, labels


def fetch_facebook_dataset():
    dataset_path = '00368/Facebook_metrics.zip'
    data_filename = 'dataset_Facebook.csv'

    original_df = fetch_uci_dataset(dataset_path, data_filename, sep=';')
    features_columns = original_df.columns[:7]
    # many columns can be chosen as label
    labels_columns = original_df.columns[7:]
    labels = original_df[labels_columns[-1]]
    labels = np.ascontiguousarray(labels, dtype=float)

    features = original_df[features_columns].values
    binarizer = FeaturesBinarizer(remove_first=True)
    features = binarizer.fit_transform(features)
    features = np.ascontiguousarray(features.toarray().astype(float))

    return features, labels


def run_solvers(model, l_l2sq, ax_list):
    solvers = []
    coeff0 = np.ones(model.n_coeffs)

    # model_dual = ModelPoisRegSDCA(l_l2sq, fit_intercept=False)
    # model_dual.fit(features, labels)
    # max_iter_dual_bfgs = 1000
    # lbfgsb_dual = LBFGSB(tol=1e-10, max_iter=max_iter_dual_bfgs,
    #                      print_every=int(max_iter_dual_bfgs / 7))
    # lbfgsb_dual.set_model(model_dual).set_prox(ProxPositive())
    # lbfgsb_dual.solve(0.2 * np.ones(model_dual.n_coeffs))
    # print(lbfgsb_dual.solution.mean())
    # print(model_dual.get_primal(lbfgsb_dual.solution))
    # for i, x in enumerate(lbfgsb_dual.history.values['x']):
    #      primal = lbfgsb._proj.call(model_dual.get_primal(x))
    #      lbfgsb_dual.history.values['obj'][i] = lbfgsb.objective(primal)
    #

    max_iter_sdca = 1000
    sdca = SDCA(l_l2sq, max_iter=max_iter_sdca,
                print_every=int(max_iter_sdca / 7), tol=1e-10)
    sdca.set_model(model).set_prox(ProxZero())
    sdca.solve()
    solvers += [sdca]

    sdca_2 = SDCA(l_l2sq, max_iter=max_iter_sdca,
                  print_every=int(max_iter_sdca / 7), tol=1e-10, batch_size=2)
    sdca_2.set_model(model).set_prox(ProxZero())
    sdca_2.solve()
    solvers += [sdca_2]

    lbfgsb = LBFGSB(max_iter=100, print_every=10, tol=1e-10)
    lbfgsb.set_model(model).set_prox(ProxL2Sq(l_l2sq, positive=True))
    lbfgsb.solve(coeff0)
    solvers += [lbfgsb]

    svrg = SVRG(max_iter=100, print_every=10, tol=1e-10, step=1e-1)
    svrg.set_model(model).set_prox(ProxL2Sq(l_l2sq, positive=True))
    svrg.solve(coeff0)
    solvers += [svrg]

    scpg = SCPG(max_iter=100, print_every=10, tol=1e-10, step=1e-3)
    scpg.set_model(model).set_prox(ProxL2Sq(l_l2sq, positive=True))
    scpg.solve(coeff0)
    solvers += [scpg]

    # print(model.n_coeffs)
    newton = Newton(max_iter=100, print_every=10)
    newton.set_model(model).set_prox(ProxL2Sq(l_l2sq))
    newton.solve(coeff0)
    solvers += [newton]

    plot_history(solvers, dist_min=True, log_scale=True,
                 x='time', ax=ax_list[0])

    ax_list[1].stem(newton.solution, linefmt='b-', markerfmt='bo', basefmt='b-')


dataset = 'crime'

if dataset == 'facebook':
    features, labels = fetch_facebook_dataset()
elif dataset == 'blog':
    features, labels = fetch_blog_dataset(n_samples=1000)
elif dataset == 'news':
    features, labels = fetch_news_popularity_dataset(n_samples=1000)
elif dataset == 'vegas':
    features, labels = fetch_las_vegas_dataset()
elif dataset == 'crime':
    features, labels = fetch_crime_dataset()
elif dataset == 'wine':
    features, labels = fetch_wine_datase()

model = ModelPoisReg(fit_intercept=False, link='identity')
model.fit(features, labels)

l_2sq_list = [1e-2, 1e-3, 1e-4, 1. / np.sqrt(len(labels))]
fig, ax_list_list = plt.subplots(2, len(l_2sq_list))
for i, l_2sq in enumerate(l_2sq_list):
    run_solvers(model, l_2sq, ax_list_list[:, i])
    ax_list_list[0, i].set_title('$\\lambda = {:.3g}$'.format(l_2sq))

plt.show()
