import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from tick.random import test_uniform, test_gaussian, test_poisson, \
    test_exponential, test_uniform_int, test_discrete, test_uniform_threaded


import sys
try:
    import statsmodels.api as sm
except ImportError:
    print("statsmodels module not found, skipping")


class QQplot:
    def __init__(self,
                 test_seed: int = 12099,
                 stat_size: int = 50000,
                 ):
        self.test_seed = test_seed
        self.stat_size = stat_size

    def randint(self,
                a: int = -2,
                b: int = 100,
                ):

        sample = test_uniform_int(a, b, self.stat_size, self.test_seed)
        fig, axs = plt.subplots(1, 1, tight_layout=True)
        axs.hist(sample, bins=b-a+1, density=True)
        fig.suptitle('uniform_int')
        return fig

    def poisson(self):
        rate = 5
        K = 20
        sample = test_poisson(rate, self.stat_size)
        fig, axs = plt.subplots(1, 1, tight_layout=True)
        axs.hist(sample, bins=K+2, range=(0, K+2), density=True)
        x = np.arange(K+2, dtype=int)
        y = np.array([stats.poisson.pmf(n, rate) for n in x], dtype=float)
        axs.plot(x, y, color='red')
        fig.suptitle('poisson')
        return fig

    def uniform(self):
        sample = test_uniform(self.stat_size, self.test_seed)
        fig = sm.qqplot(sample, stats.uniform, loc=0,
                        scale=1, fit=False, line='45')
        fig.suptitle('uniform')
        return fig

    def gaussian(self):
        sample = test_gaussian(self.stat_size, self.test_seed)
        fig = sm.qqplot(sample, stats.norm, loc=0,
                        scale=1, fit=False, line='45')
        fig.suptitle('gaussian')
        return fig

    def exponential(self):
        sample = test_exponential(1., self.stat_size, self.test_seed)
        fig = sm.qqplot(sample, stats.expon, loc=0,
                        scale=1, fit=False, line='45')
        fig.suptitle('exponential')
        return fig


def main():
    try:
        import statsmodels.api as sm
    except ImportError:
        return
    qqplot = QQplot()
    qqplot.poisson()
    qqplot.uniform()
    qqplot.gaussian()
    qqplot.exponential()
    qqplot.randint()
    plt.show()


if __name__ == '__main__':
    main()
