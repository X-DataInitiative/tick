from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import MaxNLocator
import matplotlib.pylab as plt
import numpy as np


def listdict2dictlist(LD):
    return dict(zip(LD[0], zip(*[d.values() for d in LD])))


def dictlist2listdict(DL):
    return [dict(zip(DL, t)) for t in zip(*DL.values())]


def check_feat_names(feat_names, n_features):
    if feat_names:
        if len(feat_names) != n_features:
            raise ValueError("`feat_names` sould have lenght %i" % n_features)
    else:
        feat_names = ["feature %i" % i for i in range(n_features)]
    feat_names = [n[:12] for n in feat_names]
    return feat_names


def plot_cv_scores(scores, title, figsize=None, plot_kfold_scores=False):
    scores = listdict2dictlist(scores.copy())
    scores["train"] = listdict2dictlist(scores["train"])
    scores["test"] = listdict2dictlist(scores["test"])
    n_folds = len(scores["train"]["kfold_scores"][0])
    strengths = scores["strength"]
    fig = plt.figure()
    if figsize:
        fig.set_size_inches(figsize[0], figsize[1])
    ax = plt.gca()

    def plot_report(data, colors, label):
        ax.plot(strengths, data["mean"], color=colors[0],
                label="mean " + label + " score")
        m = np.array(data["mean"])
        sd = np.array(data["sd"])
        ax.fill_between(strengths, m - sd, m + sd,
                        facecolor=colors[1], alpha=.2, edgecolor='none',
                        label=label + " score sd")
        if plot_kfold_scores:
            kf_scores = np.array(data["kfold_scores"]).T
            [ax.plot(strengths, kf_scores[i], color=colors[1])
             for i in range(n_folds)]

    plot_report(scores["train"], ["green", "lightgreen"], "train")
    plot_report(scores["test"], ["blue", "lightblue"], "test")
    ax.set_xscale("log")
    ax.set_xlabel("ProxMulti[ProxTV] strength")
    ax.set_ylabel("mean loss")
    ax.set_title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    return fig, ax


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    source: https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0.
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.

      see `plot_coefficient` function for rules defining the best values for
      start, midpoint and stop.
    '''
    __author__ = "Paul H, Horea Christian"
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def plot_coefficients(clf, figsize=None, feat_names=None):
    n_features = clf.n_features
    n_cols = clf.n_lags + 1
    feat_names = check_feat_names(feat_names, n_features)
    c = clf.coeffs.copy()
    coeffs = c.reshape((n_features, n_cols))
    vmax = c.max()
    vmin = c.min()
    m = c.mean()
    if vmax > 0 and vmin < 0:
        start = (vmax - abs(vmin)) / (2 * vmax) if m > 0 else 0
        stop = (abs(vmin) + vmax) / (2 * abs(vmin)) if m < 0 else 1
        midpoint = .5  # abs(vmin) / (vmax + abs(vmin)) if m < 0 else vmax / (vmax + abs(vmin))
        coolwarm_centered = shiftedColorMap(cmap=plt.cm.coolwarm, start=start,
                                            midpoint=midpoint, stop=stop)
    elif vmax < 0 and vmin < 0:
        stop = abs(vmax) / abs(vmin)
        coolwarm_centered = shiftedColorMap(cmap=plt.cm.Blues_r, stop=stop)
    else:
        start = abs(vmin) / abs(vmax)
        coolwarm_centered = shiftedColorMap(cmap=plt.cm.Reds, start=start)

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(coeffs, vmin=coeffs.min(), vmax=coeffs.max(),
                        cmap=coolwarm_centered, alpha=0.8)
    plt.colorbar(heatmap)

    if figsize:
        fig.set_size_inches(figsize[0], figsize[1])
    ax.set_frame_on(False)
    ax.set_ylim(0, clf.n_features)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(clf.n_features) + 0.5, minor=False)
    ax.set_yticklabels(feat_names, minor=False)

    # Turn off all the ticks
    ax.grid(False)
    ax = plt.gca()
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    plt.title("Coefficients")
    plt.xlabel("Lag")
    plt.ylabel("Feature")
    plt.tight_layout()
    return fig, ax


def intensity(x, coeffs):
    T = len(x)
    return np.exp(np.convolve(x, coeffs))[0:T]


def intensity_measure(x, coeffs):
    return np.cumsum(intensity(x, coeffs))


def plot_intensity(clf, x, intensity_func, title=None, figsize=None,
                   feat_names=None, axis_off=False, plot_id=False):
    n_features = clf.n_features
    feat_names = check_feat_names(feat_names, n_features)
    coeffs = clf.coeffs
    n_lags = clf.n_lags
    fig, axarr = plt.subplots(n_features, sharex=True, sharey=True)
    if figsize:
        fig.set_size_inches(figsize[0], figsize[1])
    axarr[-1].set_xlabel("time")
    for i in range(n_features):
        ax = axarr[i]
        if axis_off:
            ax.axis("off")
        ity = intensity_func(x,
                             coeffs[i * (n_lags + 1):(i + 1) * (n_lags + 1)])[
              0:n_lags]
        max_val = 1
        ax.step(np.arange(len(ity)), ity, color="cornflowerblue")
        ax.axhline(y=1, c="lightgrey", linestyle=":")
        id = (0, n_lags)
        if plot_id:
            ax.plot(id, id, c="lightgrey")
        m, s, b = ax.stem((x * max_val)[0:n_lags], markerfmt=".")
        plt.setp(m, 'color', 'tan', 'alpha', .7)
        plt.setp(s, 'color', 'tan', 'alpha', .3)
        plt.setp(b, 'linewidth', 0, 'alpha', .3)
        ax.set_ylabel(feat_names[i], rotation='horizontal', labelpad=30)
        ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
        ax.margins(x=0.01, y=.1)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    if title:
        axarr[0].set_title(title)
    plt.tight_layout()
    return fig, ax


def plot_probabilities(clf, x, intensity_func, figsize=None, feat_names=None,
                       axis_off=False):
    n_features = clf.n_features
    feat_names = check_feat_names(feat_names, n_features)
    coeffs = clf.coeffs
    n_lags = clf.n_lags
    fig, axarr = plt.subplots(n_features, sharex=True, sharey=True)
    if figsize:
        fig.set_size_inches(figsize[0], figsize[1])
    axarr[-1].set_xlabel("time")
    for i in range(n_features):
        ax = axarr[i]
        if axis_off:
            ax.axis("off")
        ity = intensity_func(x, coeffs[i * (n_lags + 1):(i + 1) * (n_lags + 1)])
        prob = ity / np.sum(ity)
        max_val = get_stem_max_val(x, prob)
        ax.step(np.arange(len(ity)), prob, color="cornflowerblue")
        m, s, b = ax.stem(x * max_val, markerfmt=".")
        plt.setp(m, 'color', 'tan', 'alpha', .5)
        plt.setp(s, 'color', 'tan', 'alpha', .5)
        plt.setp(b, 'linewidth', 0, 'alpha', .5)
        ax.set_ylabel(feat_names[i], rotation=90, labelpad=30)
        ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
        ax.margins(x=0.01, y=.1)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig.suptitle("Probabilities")
    plt.tight_layout()
    return fig, ax


def get_stem_max_val(x, line):
    M = np.max(x)
    m = np.min(x)
    x = (x - m) / (M - m)
    return x * np.max(line)


def report(clf, scores, path=None, figsize_coeffs=None, figsize_plots=None,
           figsize_auto=False, feat_names=None, learning_curves=True,
           coefficients=True, coefficients_stem=False,
           intensity_curves=True, cumulative_intensity_curves=False,
           probabilities=False):
    if path is None:
        callback = lambda f: plt.show(f)
    else:
        pp = PdfPages(path)
        callback = lambda f: pp.savefig(f)

    n_cols = clf.n_lags + 1
    n_features = clf.n_features
    x = np.zeros(n_cols)
    x[0] = 1

    if figsize_coeffs is None and figsize_auto:
        figsize_coeffs = (np.max([n_cols / 10, 6]), np.max([n_features / 4, 6]))

    if figsize_plots is None and figsize_auto:
        figsize_plots = (n_cols / 2, n_features / 2 * n_cols / 4)
        # (np.max([n_cols / 2, 6]), np.max([clf.n_features, 6]))

    if learning_curves:
        fig, ax = plot_cv_scores(scores, "Cross validation scores")
        callback(fig)
        plt.close(fig)

    if coefficients:
        fig, ax = plot_coefficients(clf, feat_names=feat_names,
                                    figsize=figsize_coeffs)
        callback(fig)
        plt.close(fig)

    if intensity_curves:
        fig, ax = plot_intensity(clf, x, intensity, title="Intensity",
                                 feat_names=feat_names,
                                 figsize=figsize_plots)
        callback(fig)
        plt.close(fig)

    if cumulative_intensity_curves:
        fig, ax = plot_intensity(clf, x, intensity_measure,
                                 title="Intensity_measure",
                                 figsize=figsize_plots)
        callback(fig)
        plt.close(fig)

    if probabilities:
        fig, ax = plot_probabilities(clf, x, intensity, figsize=figsize_plots,
                                     feat_names=feat_names)
        callback(fig)
        plt.close(fig)

    if path is not None:
        pp.close()
