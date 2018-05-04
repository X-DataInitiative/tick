# License: BSD 3 clause

import os
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from tick.dataset.download_helper import download_tick_dataset, get_data_home
from tick.preprocessing.features_binarizer import FeaturesBinarizer
from tick.simulation import SimuPoisReg


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
    shuffled_df = shuffle(original_df, random_state=20329)

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

    original_df = fetch_uci_dataset(dataset_path, data_filename)
    shuffled_df = shuffle(original_df, random_state=20329)

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
    shuffled_df = shuffle(original_df, random_state=20329)
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
    shuffled_df = shuffle(original_df, random_state=20329)
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
    shuffled_df = shuffle(original_df, random_state=20329)
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

    # remove_outliers = False
    # outliers = [16, 17, 51, 60, 136, 138, 143, 154, 187, 258, 349, 375, 389,
    #             417, 443, 467, 468, 499, 501, 526, 598, 599, 621, 634, 645, 652,
    #             655, 665, 667, 675, 690, 729, 758, 773, 779, 804, 814, 836, 861,
    #             880, 895, 969, 987, 1029, 1041, 1046, 1074, 1078, 1086, 1087,
    #             1106, 1114, 1173, 1176, 1180, 1188]
    # outliers = [51, 1046]
    #
    # if remove_outliers:
    #     seen_non_zeros = np.cumsum(labels != 0)
    #     # print(sum(labels != 0))
    #     # print(labels.shape)
    #     print(labels[:35])
    #     print(seen_non_zeros[:35])
    #     to_remove = np.searchsorted(seen_non_zeros - 1, outliers)
    #     print(to_remove)
    #     # labels[to_remove] = 0
    #     # print(outliers + seen_zeros[outliers])
    #     labels = np.delete(labels, to_remove)
    #     features = np.delete(features, to_remove, 0)
    #     # raise()
    #
    # print('SHAPE', len(labels), features.shape)

    labels = np.ascontiguousarray(labels)
    features = np.ascontiguousarray(features.astype(float))

    return features, labels


def fetch_property_dataset(n_samples=50999):
    # from https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction/data
    dataset_path = 'property/property.csv'
    data_filename = "property.csv"

    original_df = fetch_uci_dataset(dataset_path, data_filename, sep=',')
    shuffled_df = shuffle(original_df, random_state=20329)
    shuffled_df = shuffled_df.head(n_samples)

    labels = shuffled_df['Hazard'].values.astype(float)
    labels = np.ascontiguousarray(labels)

    features = shuffled_df.drop(['Hazard', 'Id'], axis=1).values
    binarizer = FeaturesBinarizer(remove_first=True)
    features = binarizer.fit_transform(features)
    features = np.ascontiguousarray(features.toarray().astype(float))

    return features, labels


def simulate_poisson(n_samples, n_features=None):
    if n_features is None:
        n_features = 100

    nnz = int(0.3 * n_features)

    np.random.seed(239829)

    weights = np.random.normal(size=n_features)
    mask_weights = np.random.choice(np.arange(n_features), nnz, replace=False)
    weights[mask_weights] = 0
    # weights = np.abs(weights)

    weights /= nnz
    # print(np.linalg.norm(weights) ** 2 / n_features)

    np.random.seed(2309)

    features = np.random.randn(n_samples, n_features)
    features = np.abs(features)
    features /= n_features

    epsilon = 1e-1
    while features.dot(weights).min() <= epsilon:
        n_fail = sum(features.dot(weights) <= epsilon)
        features[features.dot(weights) <= epsilon] = \
            np.random.randn(n_fail, n_features)
        features = np.abs(features)

    simu = SimuPoisReg(weights, features=features, n_samples=n_samples,
                       link='identity')
    features, labels = simu.simulate()
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


def fetch_poisson_dataset(dataset, n_samples=1000, n_features=None):
    if dataset == 'facebook':
        features, labels = fetch_facebook_dataset()
    elif dataset == 'blog':
        features, labels = fetch_blog_dataset(n_samples=n_samples)
    elif dataset == 'news':
        features, labels = fetch_news_popularity_dataset(n_samples=n_samples)
    elif dataset == 'vegas':
        features, labels = fetch_las_vegas_dataset()
    elif dataset == 'crime':
        features, labels = fetch_crime_dataset(n_samples=n_samples)
    elif dataset == 'wine':
        features, labels = fetch_wine_datase()
    elif dataset == 'property':
        features, labels = fetch_property_dataset()
    elif dataset == 'simulated':
        features, labels = simulate_poisson(n_samples, n_features=n_features)
    else:
        raise ValueError('unknown dataset {}'.format(dataset))
    return features, labels
