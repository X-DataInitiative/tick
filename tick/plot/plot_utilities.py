import matplotlib.pyplot as plt


def get_plot_color(value, n_values=11, palette='Set3'):
    try:
        from bokeh.palettes import brewer
        palette_set = brewer['Spectral']
        palette = palette_set[max(min(n_values, len(palette_set)), 3)]
        return palette[value]

    except ImportError:

        import matplotlib.cm as cmx
        import matplotlib.colors as colors

        cm = plt.get_cmap(palette)
        c_norm = colors.Normalize(vmin=0, vmax=1)
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cm)

        return scalar_map.to_rgba(value / n_values)
