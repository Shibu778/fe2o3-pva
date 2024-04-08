# This script calculates the number of atoms and volume fractions
import numpy as np

# Constants and conversion factor
density_fe2o3 = 5.24  # g/cm3
density_pva = 1.27  # g/cm3
avogadro_number = 6.02214076 * 1e23  # Number of atom in one mol
one_nm_in_cm = 1e-7

# Volume Fraction Calculation
# 20 mg fe2o3 in 20 ml pva
mass_fe2o3 = 20  # * 1e-3  # g
vol_pva = 20  # ml == cm3
vol_fe2o3 = (mass_fe2o3) / density_fe2o3  # cm3
vol_frac = vol_fe2o3 / (vol_fe2o3 + vol_pva)
print("Volume Fraction : ", vol_frac)  # 0.00019080328181644724


# Calculation of Number of atoms in 200 nm Fe2O3 particle
diameter = 200.0  # nm
radius = diameter * one_nm_in_cm / 2  # cm
volume = (4 / 3) * np.pi * radius * 3  # cm^3
mass = density_fe2o3 * volume
molar_mass_fe2o3 = 159.687  # g/mol
fe2o3_in_mol = mass / molar_mass_fe2o3  # mol
number_of_fe = fe2o3_in_mol * avogadro_number
print("Number of Fe atom : ", number_of_fe)

# Calculate the n_atom in each nanoparticle (other way)
ngrid_point = 1250000
noccup = ngrid_point * vol_frac
mass_one_nanop = mass_fe2o3 / noccup
