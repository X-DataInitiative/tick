"""
==============================================================
Study earthquake aftershocks propagation with Hawkes processes
==============================================================

This experiments aims to detect how aftershocks propagate near Japan. A Hawkes
process has been fit on data from

Ogata, Y., 1988.
Statistical models for earthquake occurrences and residual analysis
for point processes.
Journal of the American Statistical association, 83(401), pp.9-27.

On the left we can see where earthquakes have occurred and on the right
the propagation matrix, i.e. how likely a earthquake in a given zone will
trigger an aftershock in another zone.
We can observe than zone 1, 2 and 3 are tightly linked while zone 4 and
5 are more self-excited.
Note that this Hawkes process does not take the location of an earthquake
to recover this result.

Dataset `'earthquakes.csv'` is available
`here <../_static/example_data/earthquakes.csv>`_.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap

from tick.hawkes import HawkesEM
from tick.plot import plot_hawkes_kernel_norms
from tick.plot.plot_utilities import get_plot_color

df = pd.read_csv('earthquakes.csv')

lats = df.Latitude.values
lons = df.Longitude.values

# put timestamps in minutes and remove minimum
timestamps = pd.to_datetime(df.DateTime).values.astype(np.float64)
timestamps *= 1e-9 / 60
timestamps -= min(timestamps)

fig, ax_list = plt.subplots(1, 2, figsize=(8, 3.4))
ax_list[0].set_title('Earthquakes near Japan')

# Draw map
lon0, lat0 = (139, 34)
lon1, lat1 = (147, 44)
m = Basemap(projection='merc', llcrnrlon=lon0, llcrnrlat=lat0, urcrnrlon=lon1,
            urcrnrlat=lat1, resolution='l', ax=ax_list[0])
m.drawmapboundary(fill_color='#99ffff')
m.fillcontinents(color='#cc9966', lake_color='#99ffff')

# Number of splits in the map
n_groups_lats = 5

group_lats = []
group_lons = []
group_timestamps = []
group_names = []

delta_lats = (lats.max() - lats.min()) / n_groups_lats
delta_lons = lons.max() - lons.min()

for group in range(n_groups_lats):
    # Split events into groups
    min_lat_group = lats.min() + group * delta_lats
    max_lat_group = lats.min() + (group + 1) * delta_lats

    mask = (min_lat_group <= lats) & (lats < max_lat_group)

    group_lats += [lats[mask]]
    group_lons += [lons[mask]]
    group_timestamps += [timestamps[mask]]

    # Draw earthquakes on map
    x, y = m(group_lons[group], group_lats[group])
    m.scatter(x, y, 0.3, marker='o', color=get_plot_color(group))

    # Draw zone labels on map
    group_name = 'Z{}'.format(group)
    group_names += [group_name]

    center_lat = 0.5 * (min_lat_group + max_lat_group)
    x_zone, y_zone = m(lons.max() + 0.1 * delta_lons, center_lat)

    zone_bbox_style = dict(boxstyle="round", ec=(0, 0, 0, 0.9), fc=(1., 1, 1,
                                                                    0.6))
    ax_list[0].text(x_zone, y_zone, group_name, fontsize=12, fontweight='bold',
                    ha='left', va='center', color='k', withdash=True,
                    bbox=zone_bbox_style)

# Fit Hawkes process on events
events = []
for g in range(n_groups_lats):
    events += [group_timestamps[g]]
    events[g].sort()

kernel_discretization = np.linspace(0, 10000 / 60, 6)
em = HawkesEM(kernel_discretization=kernel_discretization, tol=1e-9,
              max_iter=1000, print_every=200)
em.fit(events)

plot_hawkes_kernel_norms(em, ax=ax_list[1], node_names=group_names)
ax_list[1].set_xticklabels(ax_list[1].get_xticklabels(), fontsize=11)
ax_list[1].set_yticklabels(ax_list[1].get_yticklabels(), fontsize=11)

fig.tight_layout()
plt.show()
