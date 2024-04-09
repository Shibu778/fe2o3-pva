import numpy as np
from matplotlib import pyplot as plt
import pyvista as pv


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


def plot_scalar_force(mesh, unit="(uN)", factor=1e6):
    """
    Plots the three component of force.
    """
    dargs = dict(
        scalars="force",
    )
    mesh.point_data["force"] *= factor
    pl = pv.Plotter(shape=(2, 2))
    pl.subplot(0, 0)
    pl.add_mesh(mesh, **dargs)
    pl.add_text("|F| " + unit, color="k")
    pl.subplot(0, 1)
    pl.add_mesh(mesh.copy(), component=0, **dargs)
    pl.add_text("Fx " + unit, color="k")
    pl.subplot(1, 0)
    pl.add_mesh(mesh.copy(), component=1, **dargs)
    pl.add_text("Fy " + unit, color="k")
    pl.subplot(1, 1)
    pl.add_mesh(mesh.copy(), component=2, **dargs)
    pl.add_text("Fz " + unit, color="k")
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

    molar_mass : float (in g/mol)
        molar mass of the molecule

    Return :
    number_of_molecules : int
        Number of molecules in nanoparticle
    """
    radius = diameter * one_nm_in_cm / 2  # cm
    volume = (4 / 3) * np.pi * radius * 3  # cm^3
    mass = density_fe2o3 * volume
    number_in_mol = mass / molar_mass_fe2o3  # mol
    number_of_molecule = number_in_mol * avogadro_number
    print("Number of molecule : ", number_of_molecule)
    return number_of_molecule


def mass_to_nformula_unit(mass, molar_mass=molar_mass_fe2o3):
    """
    This returns number of formula unit in mass.
    """
    return (mass / molar_mass) * avogadro_number


def plot_tot_force(
    strip,
    mag_geo=None,
    plot_magnet=False,
    scale=1,
    factor=1e6,
    unit="uN",
):
    """
    Plots the force distribution on the strip and total force vector on the center of the strip.
    Also optionally plots the magnet if necessary data are present.

    Parameters:
    ===========
    mag_geo : list (len = 6)
        Stores the magnets (start_x, width_x, start_y, width_y, start_z, width_z)
    """
    total_force = sum(strip.point_data["force"])
    direction = total_force
    cent = np.array(
        [
            ((strip.bounds[0] + strip.bounds[1]) / 2),
            ((strip.bounds[2] + strip.bounds[3]) / 2),
            ((strip.bounds[4] + strip.bounds[5]) / 2),
        ]
    )
    pl = pv.Plotter()
    pl.add_mesh(
        strip,
        scalars=strip.point_data["force"] * factor,
        scalar_bar_args={"title": "Force (" + unit + ")"},
    )
    pl.add_arrows(
        cent,
        direction,
        mag=scale,
        label="Total Force = "
        + str(np.round(np.linalg.norm(total_force) * factor, 2))
        + " "
        + unit,
        color="g",
    )
    pl.add_legend(bcolor="w", face=None)
    if plot_magnet:
        try:
            magnet = pv.Cube(
                bounds=(
                    mag_geo[0],
                    mag_geo[0] + mag_geo[1],
                    mag_geo[2],
                    mag_geo[2] + mag_geo[3],
                    mag_geo[4],
                    mag_geo[4] + mag_geo[5],
                )
            )
            pl.add_mesh(magnet, color="k")
        except:
            ValueError("Please provide proper value of magnet's geometry (mag_geo)!!")

    pl.add_axes(line_width=5)
    pl.show()


def get_center(list_of_pos):
    list_of_pos = np.array(list_of_pos)
    return sum(list_of_pos) / len(list_of_pos)


def get_div_info(strip, ndiv=10, axis_of_div=0):
    """
    Divides the given strip into ndiv, along axis_of_div.
    Returns the center of each division and the total force
    in that division.
    """
    minv = strip.points[0][axis_of_div]
    maxv = strip.points[-1][axis_of_div]
    div_array = np.linspace(minv, maxv, ndiv + 1)
    div_info = []
    for i in range(ndiv):
        indices = []
        positions = []
        info = {
            "indices": [],  # grid indices in the division
            "center": [],  # Center of the division
            "tdiv_force": [],  # Total force on division
        }
        indices = np.where(
            (strip.points[:, axis_of_div] >= div_array[i])
            & (strip.points[:, axis_of_div] <= div_array[i + 1])
        )[0]
        info["indices"] = indices
        info["center"] = get_center(strip.points[indices])
        info["tdiv_force"] = sum(strip.point_data["force"][indices])
        div_info.append(info)
    return div_info


def plot_div_force(strip, div_info, scale=1, factor=1e6, unit="uN", equal_length=False):
    """
    Plots the total force in the divided strip.

    Parameter:
    ==========
    equal_length : bool
        If true, makes all the length of the force vector equal.
    """
    pl = pv.Plotter()
    pl.add_mesh(
        strip,
        scalars=strip.point_data["force"] * factor,
        scalar_bar_args={"title": "Force on particle (" + unit + ")"},
        opacity=0.5,
    )
    center = np.array([info["center"] for info in div_info])
    tdiv_force = np.array([info["tdiv_force"] for info in div_info])
    tdiv_force_mag = np.expand_dims(np.linalg.norm(tdiv_force, axis=1), axis=1)
    if equal_length:
        tdiv_force = tdiv_force / tdiv_force_mag
        scale = 0.005
    pl.add_arrows(
        center,
        tdiv_force,
        mag=scale,
        scalar_bar_args={"title": "Force on strip division (N)"},
    )
    pl.add_axes(line_width=5)
    pl.show()
