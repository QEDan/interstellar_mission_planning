"""Constants for use in calculations"""
from scimath.units.energy import MeV
from scimath.units.length import meters as m
from scimath.units.mass import kilograms as kg
from scimath.units.time import seconds as s

G = 6.674e-11 * m**3 / kg / s**2
c = 299792458 * m / s
solar_mass = 1.98847e30 * kg
g = 9.81 * m / (s**2)
year = 3600.0 * 24.0 * 365.25 * s

# Masses were taken from wolframalpha.com
mass_3He = 2809.41 * MeV / c ** 2
mass_4He = 3728.40 * MeV / c ** 2
mass_2H = 1876.12 * MeV / c ** 2
mass_p = 938.27 * MeV / c ** 2
