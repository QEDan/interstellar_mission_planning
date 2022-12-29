"""A solar sail to accelerate a Starship"""
from typing import Optional

import numpy as np
from scimath.units.unit import unit
from scimath.units.length import astronomical_unit
from scimath.units.length import meters as m
from scimath.units.power import watt
from scimath.units.time import seconds as s

from src.constants import g, c


class SolarSail:
    """A solar sail to accelerate a Starship"""

    def __init__(self,
                 sail_mass: unit,
                 sail_radius: unit,
                 reflectivity: float = 0.9,
                 stellar_luminosity: unit = 3.83e26 * watt
                 ):
        self.sail_mass = sail_mass
        self.sail_radius = sail_radius
        self.reflectivity = reflectivity
        self.stellar_luminosity = stellar_luminosity

    def radiation_pressure_1au(self) -> unit:
        """The radiation pressure at one astronomical unit"""
        pressure = (1 + self.reflectivity) * self.stellar_luminosity / \
                   (4 * np.pi * astronomical_unit ** 2 * c)
        return pressure

    def characteristic_acceleration(self,
                                    payload_mass: unit) -> unit:
        """The characteristic acceleration of the sail

        See: Spieth&Zubrin, Ultra-Thin Solar Sails for Interstellar Travel, 1999
        http://www.niac.usra.edu/files/studies/final_report/333Christensen.pdf

        Args
            payload_mass (mass)
                Mass of the sail's payload

        Returns
            (acceleration)
                The characterisitic acceleration of the sail
        """
        total_mass = self.sail_mass + payload_mass
        characteristic_accel = self.reflectivity * self.radiation_pressure_1au() * \
            np.pi * self.sail_radius ** 2 \
            / total_mass
        return characteristic_accel

    def acceleration(self,
                     relative_position_from_star: unit,
                     payload_mass: unit,
                     max_accel: Optional[unit] = None) -> unit:
        """Returns the acceleration of the sailing starcraft

        See page 94 of Starflight Handbook, Mallove and Matloff
        Note: The constant 6.3e17 represents stellar_luminosity / (2 * c)

        Args
            relative_position_from_star (distance)
                Distance of circular sail from star
            payload_mass (mass)
                payload mass
            max_accel (acceleration)
                Maximum acceleration. Assume that the sail can be
                partly furled to limit the acceleration.
        Returns
            acceleration
                The acceleration of the sail and payload
        """
        total_mass = self.sail_mass + payload_mass
        accel = (1 + self.reflectivity) * self.stellar_luminosity * self.sail_radius ** 2 \
            / (4 * c * total_mass * relative_position_from_star ** 2)
        if max_accel:
            accel = min(accel / g, max_accel / g) * g
        accel *= np.sign(relative_position_from_star / m)
        return accel

    def final_velocity(self,
                       payload_mass: unit,
                       initial_distance_from_star: unit) -> unit:
        """Expected final velocity assuming starting from rest.

        See p. 14 of Spieth&Zubrin,
        Ultra-Thin Solar Sails for Interstellar Travel, 1999
        http://www.niac.usra.edu/files/studies/final_report/333Christensen.pdf

        Args
            payload_mass (mass)
                Mass of the payload
            initial_distance_from_star (distance)
                Initial distance from the star
        Returns
            velocity
                Expected final velocity
        """
        final_velocity = (548000 * m / s) * np.sqrt(
            self.characteristic_acceleration(payload_mass) / (m / s ** 2)
            / abs(initial_distance_from_star / astronomical_unit))
        return final_velocity
