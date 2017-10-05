import os
import pandas as pd
import numpy as np
from zipfile import ZipFile
from sklearn.utils import shuffle

from experience.poisreg_sdca import ModelPoisRegSDCA
from tick.dataset.download_helper import download_tick_dataset, get_data_home
from tick.optim.model import ModelPoisReg
from tick.optim.prox import ProxZero, ProxL2Sq, ProxPositive
from tick.optim.solver import SDCA, Newton, LBFGSB, SVRG, SCPG
from tick.plot import plot_history

BASE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/%s'

dataset_path = '00304/BlogFeedback.zip'

cache_path = os.path.join(get_data_home(), dataset_path)

if not os.path.exists(cache_path):
    cache_path = download_tick_dataset(dataset_path, base_url=BASE_URL)

zip_file = ZipFile(cache_path)

data_filename = "blogData_train.csv" #"blogData_test-2012.03.25.00_00.csv"  #

with zip_file.open(data_filename) as data_file:
    original_df = pd.read_csv(data_file, header=-1)

shuffled_df = shuffle(original_df)
n_samples = 1000

data = shuffled_df.head(n_samples).values
features = data[:, :-1]
labels = data[:, -1]
features = np.ascontiguousarray(features)
labels = np.ascontiguousarray(labels)

l_l2sq = 1e1
model = ModelPoisReg(fit_intercept=False, link='identity')
model.fit(features, labels)

print(features.shape)

model_dual = ModelPoisRegSDCA(l_l2sq, fit_intercept=False)
model_dual.fit(features, labels)
max_iter_dual_bfgs = 1000
lbfgsb_dual = LBFGSB(tol=1e-10, max_iter=max_iter_dual_bfgs,
                     print_every=int(max_iter_dual_bfgs / 7))
lbfgsb_dual.set_model(model_dual).set_prox(ProxPositive())
lbfgsb_dual.solve(0.2 * np.ones(model_dual.n_coeffs))
print(lbfgsb_dual.solution.mean())
# print(model_dual.get_primal(lbfgsb_dual.solution))

max_iter_sdca = 10000
sdca = SDCA(l_l2sq, max_iter=max_iter_sdca, print_every=int(max_iter_sdca / 7),
            tol=1e-10)
sdca.set_model(model).set_prox(ProxZero())
sdca.solve()

newton = Newton(max_iter=100, print_every=10)
newton.set_model(model).set_prox(ProxL2Sq(l_l2sq))
newton.solve(np.ones(model.n_coeffs))

lbfgsb = LBFGSB(max_iter=100, print_every=10, tol=1e-10)
lbfgsb.set_model(model).set_prox(ProxL2Sq(l_l2sq))
lbfgsb.solve(0.1 * np.ones(model.n_coeffs))

svrg = SVRG(max_iter=100, print_every=10, tol=1e-10, step=1e-5)
svrg.set_model(model).set_prox(ProxL2Sq(l_l2sq, positive=False))
svrg.solve(np.abs(sdca.solution))

scpg = SCPG(max_iter=100, print_every=10, tol=1e-10, step=1e-9)
scpg.set_model(model).set_prox(ProxL2Sq(l_l2sq, positive=False))
scpg.solve(np.abs(sdca.solution))

for i, x in enumerate(lbfgsb_dual.history.values['x']):
    primal = lbfgsb._proj.call(model_dual.get_primal(x))
    lbfgsb_dual.history.values['obj'][i] = lbfgsb.objective(primal)

plot_history([sdca, lbfgsb_dual, newton], dist_min=True, log_scale=True,
             x='time')
print(np.abs(sdca.solution).max())
