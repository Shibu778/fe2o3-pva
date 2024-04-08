# This script is used to simulate a force field
# inside a composite made up of polymer composite
# reinforced with Fe2O3 nanoparticle

# Plan
# 1. Keep everything in the first quadrant

# Imports
import magpylib as magpy
import numpy as np
import pyvista as pv
from utils import plot_B_field

# Geometry of the system
## Geometry of strip
x_width_s = 0.02  # 2 cm
y_width_s = 0.005  # 0.5 cm
z_width_s = 0.0001  # 0.1 mm

## Starting Position of strip
x_start_s = 0
y_start_s = 0.02  # 2 cm
z_start_s = 0.015  # 1.5 cm

# Find the griding in sample
x_num = 1000
y_num = 250
z_num = 5
x_grid = np.linspace(x_start_s, x_start_s + x_width_s, num=x_num)  # 100000
y_grid = np.linspace(y_start_s, y_start_s + y_width_s, num=y_num)  # 25000
z_grid = np.linspace(z_start_s, z_start_s + z_width_s, num=z_num)  # 500
s_grid = np.array(np.meshgrid(x_grid, y_grid, z_grid))
xspace = x_width_s / x_num
yspace = y_width_s / y_num
zspace = z_width_s / z_num
## Geometry of magnet
x_width_m = 0.01  # 1 cm
y_width_m = 0.005  # 0.5 cm
z_width_m = 0.03  # 3 cm

## Starting Position of magnet
x_start_m = 0 + x_width_s / 2
y_start_m = 0 + y_width_m / 2
z_start_m = 0 + z_width_m / 2


# Calculate the Magnetic Field inside the composite structure

## Magnetic moment of the commercial magnet
m_source = (0, 0, 0.45 * 5)  # In Tesla

# define the magnet
magnet = magpy.magnet.Cuboid(
    polarization=m_source,  # in SI Units (T)
    dimension=(x_width_m, y_width_m, z_width_m),  # in SI Units (m)
    position=(x_start_m, y_start_m, z_start_m),
)
print(magnet.position)

# magpy.show(magnet)
s_grid = np.swapaxes(s_grid, 0, 3)
B_s = magpy.getB(magnet, observers=s_grid)

# print(B_s.shape)
# print(B_s[0][0][0])  # Magnetic Field at coordinate
# print(s_grid[0][0][0])
# print(magpy.getB(magnet, observers=s_grid[0][0][0]))

# magpy.show(magnet)

# Plotting magnetic field on the strip
# z component
B_s = np.swapaxes(B_s, 0, 3)
B_sx = B_s[0]  # x component of the magnetic field on the strip
B_sx = np.swapaxes(B_sx, 0, 1)

B_sy = B_s[1]  # y component of the magnetic field on the strip
B_sy = np.swapaxes(B_sy, 0, 1)

B_sz = B_s[2]  # z component of the magnetic field on the strip
B_sz = np.swapaxes(B_sz, 0, 1)

strip = pv.ImageData(
    dimensions=np.array(B_sz.shape) + 1,
    spacing=(xspace, yspace, zspace),
    origin=(x_start_s, y_start_s, z_start_s),
)
strip.cell_data["B_vec"] = np.array(
    [
        [Bx, By, Bz]
        for Bx, By, Bz in zip(B_sx.flatten("F"), B_sy.flatten("F"), B_sz.flatten("F"))
    ]
)

strip.cell_data["B_mag"] = np.sqrt(
    B_sx.flatten("F") ** 2 + B_sy.flatten("F") ** 2 + B_sz.flatten("F") ** 2
)

# strip.plot(scalars="B_vec") # Magnitude
# strip.plot(scalars="B_vec", component=0)  # x-component
# strip.plot(scalars="B_vec", component=1)  # y-component
# strip.plot(scalars="B_vec", component=2)  # z-component
# plot_B_field(strip)
