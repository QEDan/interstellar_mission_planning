import numpy as np
from scimath.units.length import kilometers as km, astronomical_unit
from scimath.units.length import meters as m
from scimath.units.mass import kilograms as kg
from scimath.units.time import seconds as s

from src import SolarSail
from src.constants import g, c

solar_radius = 6.957e8 * m


class TestSolarSail:
    """Tests for the SolarSail class

    This test is based on the Icarus-1 description from Starflight Handbook
    """
    def setup_method(self):
        self.sail_mass = 0.1 * kg
        self.payload_mass = 62.9 * kg
        self.sail_radius = 1000.0 * m
        self.sail = SolarSail(self.sail_mass, self.sail_radius)

    def test_characteristic_acceleration(self):
        characteristic_accel = self.sail.characteristic_acceleration(self.payload_mass)
        assert isinstance(characteristic_accel / (m / s ** 2), float)
        assert abs(characteristic_accel / (m / s ** 2) - 0.40957) / 0.40957 < 1.0e-3

    def test_acceleration(self):
        accel = self.sail.acceleration(2 * solar_radius, self.payload_mass)
        assert isinstance(accel / (m / s ** 2), float)
        assert abs(accel - 500 * g) / (500 * g) < 1.0e-3

    def test_final_velocity(self):
        final_velocity = self.sail.final_velocity(self.payload_mass, 2 * solar_radius)
        assert isinstance(final_velocity / (m / s), float)
        assert abs(final_velocity - 0.012 * c) / (0.012 * c) < 2.0e-2

