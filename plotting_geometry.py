import pyvista as pv
import numpy as np
import magpylib as magpy

## Geometry of strip
x_width_s = 0.02  # 2 cm
y_width_s = 0.005  # 0.5 cm
z_width_s = 0.0001  # 0.1 mm

## Starting Position of strip
x_start_s = 0
y_start_s = 0.02  # 2 cm
z_start_s = 0.015  # 1.5 cm

## Geometry of magnet
x_width_m = 0.01  # 1 cm
y_width_m = 0.005  # 0.5 cm
z_width_m = 0.03  # 3 cm

## Starting Position of magnet
x_start_m = x_width_s / 2 - x_width_m / 2
y_start_m = 0 + y_width_m / 2
z_start_m = 0 + z_width_m / 2
z_start_s = z_start_s + z_start_m

x_num = 1000
y_num = 250
z_num = 5

xspace = x_width_s / x_num
yspace = y_width_s / y_num
zspace = z_width_s / z_num
strip = pv.ImageData(
    dimensions=np.array([x_num, y_num, z_num]) + 1,
    spacing=(xspace, yspace, zspace),
    origin=(x_start_s, y_start_s, z_start_s),
)

# magnet = pv.ImageData(
#     dimensions=(2, 2, 2),
#     spacing=(x_width_m, y_width_m, z_width_m),
#     origin=(x_start_m, y_start_m, z_start_m),
# )

magnet = pv.Cube(
    bounds=(
        x_start_m,
        x_start_m + x_width_m,
        y_start_m,
        y_start_m + y_width_m,
        z_start_m,
        z_start_m + z_width_m,
    )
)

pl = pv.Plotter()
pl.add_mesh(strip)
pl.add_mesh(magnet, color="#86469C")
pl.add_axes(line_width=5)
pl.show()
