# This script is used to simulate a force field
# inside a composite made up of polymer composite
# reinforced with Fe2O3 nanoparticle

# Plan
# 1. Keep everything in the first quadrant

# Imports
import magpylib as magpy
import numpy as np
import pyvista as pv
from utils import (
    plot_B_field,
    plot_glyph,
    plot_scalar_gradient,
    plot_scalar_force,
    plot_particle_contour,
    plot_particle_distr,
    calc_vol_frac,
    calc_molecule_in_nanop,
    mass_to_nformula_unit,
    plot_tot_force,
    get_div_info,
    plot_div_force,
)

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

## Starting Position of magnet center
x_start_m = 0 + x_width_s / 2
y_start_m = 0 + y_width_m / 2
z_start_m = 0 + z_width_m / 2


# Calculate the Magnetic Field inside the composite structure
# Magnetic moment of the commercial magnet
m_source = (0, 0, 0.45 * 5)  # In Tesla

# define the magnet
magnet = magpy.magnet.Cuboid(
    polarization=m_source,  # in SI Units (T)
    dimension=(x_width_m, y_width_m, z_width_m),  # in SI Units (m)
    position=(x_start_m, y_start_m, z_start_m),
)
print("Position of magnet : ", magnet.position)

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
    dimensions=np.array(B_sz.shape),
    spacing=(xspace, yspace, zspace),
    origin=(x_start_s, y_start_s, z_start_s),
)
strip.point_data["B_vec"] = np.array(
    [
        [Bx, By, Bz]
        for Bx, By, Bz in zip(B_sx.flatten("F"), B_sy.flatten("F"), B_sz.flatten("F"))
    ]
)

strip.point_data["B_mag"] = np.sqrt(
    B_sx.flatten("F") ** 2 + B_sy.flatten("F") ** 2 + B_sz.flatten("F") ** 2
)

# Get the grid positions in flatten array
s_grid = np.swapaxes(s_grid, 0, 3)
s_gridx = s_grid[0]
s_gridx = np.swapaxes(s_gridx, 0, 1)

s_gridy = s_grid[1]
s_gridy = np.swapaxes(s_gridy, 0, 1)

s_gridz = s_grid[2]
s_gridz = np.swapaxes(s_gridz, 0, 1)

strip.point_data["pos"] = np.array(
    [
        [x, y, z]
        for x, y, z in zip(
            s_gridx.flatten("F"), s_gridy.flatten("F"), s_gridz.flatten("F")
        )
    ]
)

# strip.plot(scalars="B_vec")  # Magnitude
# strip.plot(scalars="B_vec", component=0)  # x-component
# strip.plot(scalars="B_vec", component=1)  # y-component
# strip.plot(scalars="B_vec", component=2)  # z-component
# plot_B_field(strip)
# plot_glyph(strip)  # Very expensive


# Computing the gradient (Scalar field gradient)
mesh_g = plot_scalar_gradient(strip)
strip["grad_B"] = mesh_g["gradient"]

# Force calculations
np.random.seed(6)
mag_mom_per_Fe = [3, 4]
cos_theta_range = [-1, 1]
# mu_cos_theta = gen_mu_cos_theta()
bohr_mag_to_J_per_T = 9.274009994 * 1e-24

vol_frac_nano = calc_vol_frac(20, 20)
# Total number of Grid point and number to be occupied by nanoparticles
ngrid_point = len(strip.point_data["pos"])  # number of grid point
noccup = int(
    ngrid_point * vol_frac_nano
)  # Total number of points to be occupied by nanoparticle

# Let's consider 200 nm has 100 Fe atom
mass_per_particle = 20 / noccup
natom_in_particle = 2 * mass_to_nformula_unit(mass_per_particle)
mu_range = np.array(mag_mom_per_Fe) * natom_in_particle  # bohr-magneton
n_div = 101

# Generate random grid points to occupy
gridp_to_occupy = np.random.randint(0, ngrid_point, size=noccup)

# Select mu values
mu_array = np.linspace(
    mu_range[0], mu_range[1], n_div
)  # Possible mu values to select from
mu_for_nano = np.random.choice(mu_array, size=noccup)

# Select cos_theta
cos_theta_array = np.linspace(cos_theta_range[0], cos_theta_range[1], n_div)
cos_theta_nano = np.random.choice(cos_theta_array, size=noccup)

# Calculate mu * cos_theta
mu_cos_theta_nano = mu_for_nano * cos_theta_nano

# Make a factor array
factor = np.array([0.0] * ngrid_point)
for i in range(noccup):
    factor[gridp_to_occupy[i]] = mu_cos_theta_nano[i]

strip.point_data["mu_cos_theta"] = factor
factor = np.expand_dims(factor, 1)
strip.point_data["force"] = factor * strip.point_data["grad_B"] * bohr_mag_to_J_per_T

# Plot force magnitude, x, y and z components
# plot_scalar_force(strip)

# Plot the particle distribution
# plot_particle_contour(strip)
# plot_particle_distr(strip, gridp_to_occupy, 10)

# mag_geo = [x_start_m, x_width_m, y_start_m, y_width_m, z_start_m, z_width_m]
# plot_tot_force(strip, mag_geo=mag_geo, plot_magnet=False)
div_info = get_div_info(strip)
# print(div_info)
plot_div_force(strip, div_info, scale=1)
