from scipy import stats
import numpy as np


def randint(
        a=-2,
        b=100,
        sample_size=10000,
):
    sample = stats.randint(low=a, high=b).rvs(size=sample_size)
    probs = (1. / (b-a)) * np.ones(shape=(b-a,))
    f_exp = sample_size * probs
    f_obs, _ = np.histogram(sample, bins=range(a, 1+b))
    assert f_obs.shape == f_exp.shape
    assert np.allclose(np.sum(f_obs), np.sum(f_exp), rtol=1e-8, atol=1e-16)
    res = stats.chisquare(f_obs=f_obs, f_exp=f_exp)
    print("\nChi square test of   randint sample from scipy.stats")
    print(f"a: {a}")
    print(f"b: {b}")
    print(f"sample_size: {sample_size}")
    print(f"results: {res}\n")
    return res


def uniform(
        sample_size=10000,
):
    sample = stats.uniform().rvs(size=sample_size)
    res = stats.kstest(sample, 'uniform')
    print("\nKolmogorov-smirnov test of uniform sample from scipy.stats")
    print(f"sample_size: {sample_size}")
    print(f"results: {res}\n")
    return res


def gaussian(
        sample_size=10000,
):
    sample = stats.norm.rvs(size=sample_size)
    res = stats.kstest(sample, 'norm')
    print("\nKolmogorov-smirnov test of gaussian sample from scipy.stats")
    print(f"sample_size: {sample_size}")
    print(f"results: {res}\n")
    return res


def poisson(
        rate=5,
        sample_size=10000,
        K=20,
):
    sample = stats.poisson(rate).rvs(size=sample_size)

    # To test statistical consistency of poisson we do like if it was a
    # discrete law with a probability of sum_{k>K}(P(k)) for the last event
    probs_ = [
        stats.poisson.pmf(i, rate) for i in range(K)
    ]
    obs_ = [sum(sample == i) for i in range(K)]

    # We add the last event
    obs_.append(sum(sample >= K))
    probs_.append(1 - sum(probs_))

    f_exp = sample_size * np.array(probs_, dtype=float)
    f_obs = np.array(obs_, dtype=float)
    res = stats.chisquare(f_exp=f_exp, f_obs=f_obs)
    print("\nChi square test of Poisson random sample from scipy.stats")
    print(f"rate: {rate}")
    print(f"sample_size: {sample_size}")
    print(f"K: {K}")
    print(f"results: {res}\n")
    return res


def expon(
        intensity=1.,
        sample_size=10000,
):
    sample = stats.expon(scale=1./intensity).rvs(size=sample_size)
    res = stats.kstest(sample, 'expon', (0, 1./intensity))
    print("\nKolmogorov-Smirnov test of exponential random sample from scipy.stats")
    print(f"intensity: {intensity}")
    print(f"sample_size: {sample_size}")
    print(f"results: {res}\n")
    return res


def main():
    randint()
    uniform()
    gaussian()
    poisson()
    expon()


if __name__ == '__main__':
    main()
