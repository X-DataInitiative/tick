import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tick.online import OnlineForestClassifier

from bokeh.models.glyphs import Circle, Segment, Text
from bokeh.io import curdoc
from bokeh.layouts import row, column, widgetbox
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure

import pandas as pd

n_samples = 500
n_features = 2
max_iter = 50

X, y = make_classification(n_samples=n_samples, n_features=n_features,
                           n_informative=n_features, n_redundant=0)

X = MinMaxScaler().fit_transform(X)

n_classes = int(y.max() + 1)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, stratify=y, test_size=.3, random_state=42)


def get_tree(of):
    df = of.get_nodes_df(0)
    df.sort_values(by=['depth', 'parent', 'id'], inplace=True)
    # max_depth = df.depth.max()
    max_depth = 10
    n_nodes = df.shape[0]
    x = np.zeros(n_nodes)
    x[0] = 0.5
    indexes = df['id'].as_matrix()
    df['x'] = x
    df['y'] = max_depth - df['depth']
    df['x0'] = df['x']
    df['y0'] = df['y']
    for node in range(1, n_nodes):
        index = indexes[node]
        parent = df.at[index, 'parent']
        depth = df.at[index, 'depth']
        left_parent = df.at[parent, 'left']
        x_parent = df.at[parent, 'x']
        if left_parent == index:
            # It's a left node
            df.at[index, 'x'] = x_parent - 0.5 ** (depth + 1)
        else:
            df.at[index, 'x'] = x_parent + 0.5 ** (depth + 1)
        df.at[index, 'x0'] = x_parent
        df.at[index, 'y0'] = df.at[parent, 'y']

    df['color'] = df['leaf'].astype('str')
    df.replace({'color': {'False': 'blue', 'True': 'green'}}, inplace=True)
    return df


of = OnlineForestClassifier(n_classes=n_classes, seed=1234,
                            use_aggregation=False,
                            n_trees=1, split_pure=True,
                            dirichlet=0.5, step=1.,
                            use_feature_importances=False)

dfs = {}
df_datas = {}

for t in range(0, max_iter + 1):
    of.partial_fit(X[t].reshape(1, n_features), np.array([y[t]]))
    dfs[t] = get_tree(of)
    df_datas[t] = pd.DataFrame({'x1': X[:(t+1), 0],
                                'x2': X[:(t+1), 1],
                                'y': y[:(t+1)]},
                               columns=['x1', 'x2', 'y'])

df = dfs[1]
df_data = df_datas[1]

# plot_options = dict(plot_width=700, plot_height=400,
#                     outline_line_color=None)
# ydr = DataRange1d(range_padding=0.05)
# xdr = DataRange1d(range_padding=0.05)

# plot = Plot(x_range=xdr, y_range=ydr, **plot_options)

source = ColumnDataSource(ColumnDataSource.from_df(df))
source_data = ColumnDataSource(ColumnDataSource.from_df(df_data))

plot = figure(plot_width=400, plot_height=400, title="Mondrian Tree",
              tools="",
              # tools="pan,reset,save,wheel_zoom",
              x_range=[-0.1, 1.1], y_range=[-1, 11])

plot_data = figure(plot_width=400, plot_height=400, title="Data points",
                   tools="",
                   x_range=[0.1, 1.1], y_range=[0.1, 1.1])

plot.outline_line_color = None
plot.axis.visible = False
plot.grid.visible = False

plot_data.outline_line_color = None
plot_data.grid.visible = None

# plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

# text_glyph = Text(x="x", y="y", text="index")
#
# plot.add_glyph(source, text_glyph)

# circles = Circle(x="x", y="y", size=10, fill_color="color",
#                  name="circles", fill_alpha=0.1)
# cir = plot.add_glyph(source, circles)


circles = plot.circle(x="x", y="y", size=10, fill_color="color", name="circles",
                      fill_alpha=0.1, source=source)


circles_data = plot_data.circle(x="x1", y="x2", size=10,
                                fill_color="y",
                                name="circles",
                                fill_alpha=0.5, source=source_data)


# TODO: add the decision function using bokeh image. Precompute everything of course...
# import numpy as np
#
# from bokeh.plotting import figure, show, output_file
# from bokeh.models import HoverTool
#
# N = 500
# x = np.linspace(0, 10, N)
# y = np.linspace(0, 10, N)
# xx, yy = np.meshgrid(x, y)
# d = np.sin(xx)*np.cos(yy)
#
# p = figure(x_range=(0, 10), y_range=(0, 10),
#            tools=[HoverTool(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])])
#
# # must give a vector of image data for image parameter
# p.image(image=[d], x=0, y=0, dw=10, dh=10, palette="Spectral11")


hover = HoverTool(
    renderers=[circles],
    tooltips=[
        ("time", "@time"),
        ("threshold", "@threshold")
    ]
)

plot.add_tools(hover)

plot.text(x="x", y="y", text="id", source=source)

# plot.circle(x="x", y="y", size=10, fill_color="color", name="circles",
#             fill_alpha=0.1, source=source)

plot.segment(x0="x", y0="y", x1="x0", y1="y0", line_color="#151515",
             line_alpha=0.4, source=source)


def update_plot(attrname, old, new):
    t = iteration_slider.value
    source.data = dfs[t].to_dict('list')
    source_data.data = df_datas[t].to_dict('list')
    # print(df)


iteration_slider = Slider(title="Iteration", value=0, start=0,
                          end=max_iter, step=1, orientation='horizontal')

iteration_slider.on_change('value', update_plot)

inputs = widgetbox(iteration_slider, sizing_mode='scale_both')

curdoc().add_root(row(inputs, plot, plot_data, width=1200, height=500))

# curdoc().add_root(column(plot, inputs, width=800))

curdoc().title = "Mondrian trees"
