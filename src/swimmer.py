"""A SWIMMER engine to accelerate a starship.

This engine is propellantless, requiring only electrical power
which can be beamed or produced on board.

Reference:
    Spacecraft With Interstellar Medium
    Momentum Exchange Reactions: The potential and limitations
    of propellantless interstellar travel
    By Drew Brisbin
    https://arxiv.org/abs/1808.02019
"""

import numpy as np
from scimath.units import unit
from scimath.units.length import meters as m
from scimath.units.time import seconds as s
from scimath.units.mass import kilograms as kg


class Swimmer:
    """A SWIMMER engine to accelerate a starship."""

    def __init__(self,
                 pusher_area: unit,
                 pusher_areal_density: unit = 1.0 * kg / m ** 2,
                 ion_mass: unit = 1.67262192369e-27 * kg,  # proton mass
                 ion_density: unit = 0.07 / (m / 100) ** 3,  # ISM proton density
                 ):
        self.pusher_area = pusher_area
        self.pusher_areal_density = pusher_areal_density
        self.ion_mass = ion_mass
        self.ion_density = ion_density

    def shed_area(self,
                  delta_area: unit):
        """Remove specified amount of area from the pusher plate to shed mass"""
        self.pusher_area -= delta_area

    def pusher_mass(self) -> unit:
        """return the mass of the pusher plate"""
        return self.pusher_area * self.pusher_areal_density

    def acceleration(self,
                     power_delivered: unit,
                     abs_velocity: unit,
                     total_mass: unit,
                     braking: bool = False) -> unit:
        """Compute the acceleration from the SWIMMER engine."""
        sign = -1 if braking else 1
        exposure = self.pusher_area * self.ion_mass * self.ion_density * abs_velocity
        force_unit = kg * m / s ** 2
        force = sign * np.sqrt(
            exposure * (2 * power_delivered + exposure * abs_velocity ** 2) / (force_unit ** 2)
        ) * force_unit - exposure * abs_velocity
        acceleration = force / total_mass
        return acceleration
