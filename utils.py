import numpy as np
from matplotlib import pyplot as plt
import pyvista as pv


def dipole(m, r, r0):
    """Calculate a field in point r created by a dipole moment m located in r0.
    Spatial components are the outermost axis of r and returned B.

    Reference : https://michal.rawlik.pl/2015/03/12/magnetic-dipole-in-python/

    Parameters:
    ===========
    m : list or np.array
        x, y and z component of magnetic moment

    r : list or np.array
        x, y, z coordinate of point or points where we need the value of B

    r0 : list or np.array
        x, y, z coordinate of point where the magnetic dipole lies

    Returns:
    ========
    B : np.array
        x, y and z component of magnetic field vector at given r

    """
    # we use np.subtract to allow r and r0 to be a python lists, not only np.array
    R = np.subtract(np.transpose(r), r0).T

    # assume that the spatial components of r are the outermost axis
    norm_R = np.sqrt(np.einsum("i...,i...", R, R))

    # calculate the dot product only for the outermost axis,
    # that is the spatial components
    m_dot_R = np.tensordot(m, R, axes=1)

    # tensordot with axes=0 does a general outer product - we want no sum
    B = 3 * m_dot_R * R / norm_R**5 - np.tensordot(m, 1 / norm_R**3, axes=0)

    # include the physical constant
    B *= 1e-7

    return B


# Use
# 3D
# dipole(m=[1, 2, 3], r=[1, 1, 2], r0=[0, 0, 0])

# 2D

# X = np.linspace(-1, 1)
# Y = np.linspace(-1, 1)

# Bx, By = dipole(m=[0, 0.1], r=np.meshgrid(X, Y), r0=[-0.2,0.1])

# plt.figure(figsize=(8, 8))
# plt.streamplot(X, Y, Bx, By)
# plt.margins(0, 0)


def plot_B_field(mesh):
    """
    Plots the Vector field in a given object
    """
    dargs = dict(
        scalars="B_vec",
    )
    pl = pv.Plotter(shape=(2, 2))
    pl.subplot(0, 0)
    pl.add_mesh(mesh, **dargs)
    pl.add_text("|B| (T)", color="k")
    pl.subplot(0, 1)
    pl.add_mesh(mesh.copy(), component=0, **dargs)
    pl.add_text("Bx (T)", color="k")
    pl.subplot(1, 0)
    pl.add_mesh(mesh.copy(), component=1, **dargs)
    pl.add_text("By (T)", color="k")
    pl.subplot(1, 1)
    pl.add_mesh(mesh.copy(), component=2, **dargs)
    pl.add_text("Bz (T)", color="k")
    pl.link_views()
    pl.camera_position = "iso"
    pl.background_color = "white"
    pl.show()


def plot_glyph(mesh, key="B_vec"):
    """
    Plot glyphs or arrows for vectors.
    """
    mesh.set_active_vectors(key)
    p = pv.Plotter()
    p.add_mesh(mesh.arrows, lighting=False, scalar_bar_args={"title": "B Magnitude"})
    p.add_mesh(mesh, color="grey", ambient=0.6, opacity=0.5, show_edges=False)
    p.show()


def gradients_to_dict(mesh_g, arr):
    """A helper method to label the gradients into a dictionary."""
    keys = np.array(
        [
            "du/dx",
            "du/dy",
            "du/dz",
            "dv/dx",
            "dv/dy",
            "dv/dz",
            "dw/dx",
            "dw/dy",
            "dw/dz",
        ]
    )
    keys = keys.reshape((3, 3))[:, : arr.shape[1]].ravel()
    return dict(zip(keys, mesh_g["gradient"].T))


def plot_scalar_gradient(strip, key="B_mag", plot=False):
    """
    Plots the three component of gradient of a scalar field.
    """
    # Computing the gradient (Scalar field gradient)
    strip = strip
    mesh_g = strip.compute_derivative(scalars=key)
    gradients = gradients_to_dict(mesh_g, mesh_g["gradient"])
    mesh_g.point_data.update(gradients)

    keys = np.array(list(gradients.keys())).reshape(1, 3)

    p = pv.Plotter(shape=keys.shape)

    for (i, j), name in np.ndenumerate(keys):
        name = keys[i, j]
        p.subplot(i, j)
        p.add_mesh(mesh_g.contour(scalars=name), scalars=name, opacity=0.75)
        p.add_mesh(mesh_g.outline(), color="k")
    p.link_views()
    p.view_isometric()
    if plot:
        p.show()
    return mesh_g


def plot_scalar_force(mesh):
    """
    Plots the three component of force.
    """
    dargs = dict(
        scalars="force",
    )
    pl = pv.Plotter(shape=(2, 2))
    pl.subplot(0, 0)
    pl.add_mesh(mesh, **dargs)
    pl.add_text("|F| (N)", color="k")
    pl.subplot(0, 1)
    pl.add_mesh(mesh.copy(), component=0, **dargs)
    pl.add_text("Fx (N)", color="k")
    pl.subplot(1, 0)
    pl.add_mesh(mesh.copy(), component=1, **dargs)
    pl.add_text("Fy (N)", color="k")
    pl.subplot(1, 1)
    pl.add_mesh(mesh.copy(), component=2, **dargs)
    pl.add_text("Fz (N)", color="k")
    pl.link_views()
    pl.camera_position = "iso"
    pl.background_color = "white"
    pl.show()


def plot_particle_contour(strip):
    """
    Plots the particle distribution. Takes mu_cos_theta
    values to assign colors.
    """
    p = pv.Plotter()
    p.add_mesh(
        strip.contour(scalars="mu_cos_theta"), scalars="mu_cos_theta", opacity=0.75
    )
    p.add_mesh(strip.outline(), color="k")
    p.show()


def plot_particle_distr(strip, gridp_to_occupy, point_size=1):
    """
    Plots particle distribution.

    strip : pv.ImageData
        Represents the composite strip
    gridp_to_occupy : np.array
        Stores indices of the flatten array containing
        positions of grid points of strip
    """
    grid_pos_to_occupy = strip.points[gridp_to_occupy]
    p = pv.Plotter()
    p.add_points(
        grid_pos_to_occupy,
        style="points",
        render_points_as_spheres=True,
        point_size=point_size,
        color="k",
    )
    p.add_mesh(strip.outline(), color="g")
    p.show()


# Constants and conversion factor
density_fe2o3 = 5.24  # g/cm3
density_pva = 1.27  # g/cm3
avogadro_number = 6.02214076 * 1e23  # Number of atom in one mol
one_nm_in_cm = 1e-7
molar_mass_fe2o3 = 159.687  # g/mol


def calc_vol_frac(r_weight, m_vol):
    """
    Calculates volume fraction of nanoparticle,
    given its mass and volume of matrix in composite.

    Polymer matrix may shrink during composite formation.
    The shrinking effect is not considered in this volume
    fraction calculation.

    Parameters:
    ===========
    r_mass : float (in gram or g)
        total weight of the nanoparticles in composite
    m_volume : float (in ml or cm^3)
        total volume of the matrix in composite

    Results:
    ========
    vol_frac : float
        Volume fraction of nanoparticle in composite
    """
    mass_fe2o3 = r_weight  # in g
    vol_pva = m_vol  # in ml or cm3, both are the same
    vol_fe2o3 = (mass_fe2o3) / density_fe2o3  # cm3
    vol_frac = vol_fe2o3 / (vol_fe2o3 + vol_pva)
    print("Volume Fraction of nanoparticle : ", vol_frac)
    return vol_frac


def calc_molecule_in_nanop(diameter, molar_mass=molar_mass_fe2o3):
    """
    This function calculates the number of molecule of
    compound in a nanoparticle of given radius.

    Parameter :
    ===========
    diameter : float (in nm)
        diameter of the nanoparticle in nanometer

    """
    radius = diameter * one_nm_in_cm / 2  # cm
    volume = (4 / 3) * np.pi * radius * 3  # cm^3
    mass = density_fe2o3 * volume
    number_in_mol = mass / molar_mass_fe2o3  # mol
    number_of_molecule = number_in_mol * avogadro_number
    print("Number of molecule : ", number_of_molecule)
    return number_of_molecule
