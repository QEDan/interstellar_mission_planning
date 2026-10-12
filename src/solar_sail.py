"""A solar sail to accelerate a Starship"""
from typing import Optional

import numpy as np
from scimath.units.unit import unit
from scimath.units.length import astronomical_unit
from scimath.units.length import meters as m
from scimath.units.power import watt
from scimath.units.time import seconds as s

from src.constants import g, c, G, solar_mass
from src.utils import v_escape


class SolarSail:
    """A solar sail to accelerate a Starship"""

    def __init__(self,
                 sail_mass: unit,
                 sail_radius: unit,
                 *,
                 reflectivity: float = 0.9,
                 stellar_luminosity: unit = 3.83e26 * watt,
                 star_mass: unit = solar_mass
                 ):
        self.sail_mass = sail_mass
        self.sail_radius = sail_radius
        self.reflectivity = reflectivity
        self.stellar_luminosity = stellar_luminosity
        self.star_mass = star_mass

    def radiation_pressure_1au(self,
                               stellar_luminosity: Optional[unit] = None) -> unit:
        """The radiation pressure at one astronomical unit"""
        if stellar_luminosity is None:
            stellar_luminosity = self.stellar_luminosity
        pressure = (1 + self.reflectivity) * stellar_luminosity / \
                   (4 * np.pi * astronomical_unit ** 2 * c)
        return pressure

    def characteristic_acceleration(self,
                                    payload_mass: unit,
                                    stellar_luminosity: Optional[unit] = None) -> unit:
        """The characteristic acceleration of the sail

        See: Spieth&Zubrin, Ultra-Thin Solar Sails for Interstellar Travel, 1999
        http://www.niac.usra.edu/files/studies/final_report/333Christensen.pdf

        Args
            payload_mass (mass)
                Mass of the sail's payload
            stellar_luminosity (power)
                Luminosity of the star. Defaults to self.stellar_luminosity.

        Returns
            (acceleration)
                The characterisitic acceleration of the sail
        """
        total_mass = self.sail_mass + payload_mass
        rad_p = self.radiation_pressure_1au(stellar_luminosity)
        accel = self.reflectivity * rad_p * np.pi * self.sail_radius ** 2 / total_mass
        return accel

    def acceleration(self,
                     relative_position_from_star: unit,
                     payload_mass: unit,
                     *,
                     max_accel: Optional[unit] = None,
                     star_mass: Optional[unit] = None,
                     stellar_luminosity: Optional[unit] = None,
                     include_gravity: bool = False) -> unit:
        """Returns the acceleration of the sailing starcraft.

        See page 94 of Starflight Handbook, Mallove and Matloff

        Args
            relative_position_from_star (distance)
                Distance of circular sail from star
            payload_mass (mass)
                payload mass
            max_accel (acceleration)
                Maximum acceleration. Assume that the sail can be
                partly furled to limit the acceleration.
            star_mass (mass)
                Mass of the star. Defaults to self.star_mass.
            stellar_luminosity (power)
                Luminosity of the star. Defaults to self.stellar_luminosity.
            include_gravity (bool)
                Whether to subtract stellar gravitational acceleration.
        Returns
            acceleration
                The acceleration of the sail and payload
        """
        if star_mass is None:
            star_mass = self.star_mass
        if stellar_luminosity is None:
            stellar_luminosity = self.stellar_luminosity
        total_mass = self.sail_mass + payload_mass
        rad_accel = (1 + self.reflectivity) * stellar_luminosity * self.sail_radius ** 2 \
            / (4 * c * total_mass * relative_position_from_star ** 2)
        if include_gravity:
            grav_accel = G * star_mass / (relative_position_from_star ** 2)
            accel = rad_accel - grav_accel
        else:
            accel = rad_accel
        if max_accel:
            accel = min(accel / g, max_accel / g) * g
        accel *= np.sign(relative_position_from_star / m)
        return accel

    def final_velocity(self,
                       payload_mass: unit,
                       initial_distance_from_star: unit,
                       *,
                       star_mass: Optional[unit] = None,
                       stellar_luminosity: Optional[unit] = None,
                       include_gravity: bool = False) -> unit:
        """Expected final velocity assuming starting from rest.

        See p. 14 of Spieth&Zubrin,
        Ultra-Thin Solar Sails for Interstellar Travel, 1999
        http://www.niac.usra.edu/files/studies/final_report/333Christensen.pdf

        Args
            payload_mass (mass)
                Mass of the payload
            initial_distance_from_star (distance)
                Initial distance from the star
            star_mass (mass)
                Mass of the star. Defaults to self.star_mass.
            stellar_luminosity (power)
                Luminosity of the star. Defaults to self.stellar_luminosity.
            include_gravity (bool)
                Whether to subtract stellar escape velocity.
        Returns
            velocity
                Expected final velocity
        """
        if star_mass is None:
            star_mass = self.star_mass
        dist = abs(initial_distance_from_star / astronomical_unit)
        if dist == 0:
            dist = 1.0
        v_rad = (548000 * m / s) * np.sqrt(
            self.characteristic_acceleration(payload_mass, stellar_luminosity) / (m / s ** 2)
            / dist)

        if include_gravity:
            v_esc = v_escape(abs(initial_distance_from_star), star_mass)
            v_rad_sq = (v_rad / (m / s)) ** 2
            v_esc_sq = (v_esc / (m / s)) ** 2
            net_v_sq = max(0.0, v_rad_sq - v_esc_sq)
            return np.sqrt(net_v_sq) * (m / s)
        return v_rad
