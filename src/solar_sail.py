"""A solar sail to accelerate a Starship"""

import numpy as np
from scimath.units.length import astronomical_unit
from scimath.units.length import meters as m
from scimath.units.mass import kilograms as kg
from scimath.units.time import seconds as s
from src.constants import g


class SolarSail:
    """A solar sail to accelerate a Starship.


    """

    def __init__(self,
                 sail_mass,
                 sail_radius,
                 reflectivity=0.9
                 ):
        self.sail_mass = sail_mass
        self.sail_radius = sail_radius
        self.reflectivity = reflectivity

    def characteristic_acceleration(self,
                                    payload_mass):
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
        characteristic_accel = self.reflectivity * (9.126e-6 * m / s ** 2) * \
            np.pi * (self.sail_radius / m) ** 2 \
            / (total_mass / kg)
        return characteristic_accel

    def acceleration(self,
                     distance_from_star,
                     payload_mass,
                     max_accel=None):
        """Returns the acceleration of the sailing starcraft

        See page 94 of Starflight Handbook, Mallove and Matloff

        Args
            distance_from_star (distance)
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
        accel = (1 + self.reflectivity) * 6.3e17 * (self.sail_radius / m) ** 2 \
            / (2 * (total_mass / kg) * (distance_from_star / m) ** 2) * (m / s ** 2)
        if max_accel:
            accel = min(accel / g, max_accel / g) * g
        return accel

    def final_velocity(self, payload_mass, initial_distance_from_star):
        """Expected final velocity assuming starting from rest.

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
            / (initial_distance_from_star / astronomical_unit))
        return final_velocity
