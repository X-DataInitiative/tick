import os

from tick.array.serialize import tick_double_sparse2d_from_file, \
    tick_double_array_from_file

from tick.linear_model.model_logreg import ModelLogReg
from tick.prox.prox_elasticnet import ProxElasticNet
from tick.solver.saga import SAGA

# Create this dataset with benchmark_util
dirpath = os.path.dirname(__file__)
features_path = os.path.join(dirpath, "data", "url.3.features.cereal")
labels_path = os.path.join(dirpath, "data", "url.3.labels.cereal")

N_ITER = 200
n_samples = 196000
ALPHA = 1. / n_samples
BETA = 1e-10
STRENGTH = ALPHA + BETA
RATIO = BETA / STRENGTH
THREADS = 8

features = tick_double_sparse2d_from_file(features_path)
labels = tick_double_array_from_file(labels_path)
model = ModelLogReg().fit(features, labels)
prox = ProxElasticNet(STRENGTH, RATIO)
saga = SAGA(
    max_iter=N_ITER,
    tol=0,
    rand_type="unif",
    step=0.00257480411965,
    n_threads=THREADS,
    verbose=False,
    record_every=20,
)
saga.history.print_order += ['time']
saga.set_model(model).set_prox(prox)
saga.solve()
saga.print_history()
