from scipy import stats
import numpy as np


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
    print("Chi square test of Poisson random sample from scipy.stats")
    print(f"rate: {rate}")
    print(f"sample_size: {sample_size}")
    print(f"K: {K}")
    print(f"results: {res}")
    return res


def expon(
        intensity=1.,
        sample_size=10000,
):
    sample = stats.expon(scale=1./intensity).rvs(size=sample_size)
    res = stats.kstest(sample, 'expon', (0, 1./intensity))
    print("Kolmogorov-Smirnov test of exponential random sample from scipy.stats")
    print(f"intensity: {intensity}")
    print(f"sample_size: {sample_size}")
    print(f"results: {res}")
    return res


def main():
    poisson()
    expon()


if __name__ == '__main__':
    main()
