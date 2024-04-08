# Magnetic Field of spherical magnet

import pymagnet as pm
import numpy as np


def gen_sphere(alpha=0, beta=0, gamma=0, **kwargs):
    # pm.reset()

    radius = 10
    hGap = kwargs.pop("hGap", 1)
    mask_magnet = kwargs.pop("mask_magnet", True)

    center = (0, 0, 0)
    m1 = pm.magnets.Sphere(
        radius=radius,
        Jr=1.0,
        center=center,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        mask_magnet=mask_magnet,
    )

    return m1


mask_magnet = False  # mask values with NaN inside a magnet
show_magnets = True  # draw magnet in plots

m1 = gen_sphere(alpha=0, beta=0, gamma=0, mask_magnet=mask_magnet)


fig_slice, slice_cache, data_objects = pm.plots.slice_quickplot(
    cmax=0.5,
    num_levels=11,
    opacity=1.0,
    num_arrows=10,
    num_points=200,
    cone_opacity=0.9,
    magnet_opacity=1.0,
    mask_magnet=mask_magnet,
    show_magnets=show_magnets,
    colorscale="viridis",
    max1=20,
    max2=20,
    slice_value=0.0,
    unit="mm",
)

# for plane in slice_cache.keys():
#     pm.plots.plot_3D_contour(
#         slice_cache[plane]["points"],
#         slice_cache[plane]["field"],
#         plane,
#         cmin=0,
#         cmax=0.5,
#         num_levels=11,
#         #                              num_arrows = 21,
#         #                              vector_color = 'k'
#     )

for plane in slice_cache.keys():
    pm.plots.plot_3D_contour(
        slice_cache[plane]["points"],
        slice_cache[plane]["field"],
        plane,
        cmin=0,
        cmax=0.5,
        num_levels=6,
        plot_type="streamplot",
        cmap="copper",
        stream_color="normal",
    )
