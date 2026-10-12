import pytest
from scimath.units.length import meters as m
from scimath.units.mass import kilograms as kg
from scimath.units.power import watt
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
        assert abs(characteristic_accel / (m / s ** 2) - 0.38737) / 0.38737 < 1.0e-3

    def test_acceleration(self):
        accel = self.sail.acceleration(2 * solar_radius, self.payload_mass)
        assert isinstance(accel / (m / s ** 2), float)
        assert abs(accel - 507.2 * g) / (507.2 * g) < 1.0e-3

    def test_max_acceleration(self):
        max_accel = 0.01 * g
        accel = self.sail.acceleration(2 * solar_radius,
                                       self.payload_mass,
                                       max_accel=max_accel)
        assert isinstance(accel / (m / s ** 2), float)
        assert abs(accel - max_accel) / (max_accel) < 1.0e-3

    def test_final_velocity(self):
        final_velocity = self.sail.final_velocity(self.payload_mass, 2 * solar_radius)
        assert isinstance(final_velocity / (m / s), float)
        assert abs(final_velocity - 0.012 * c) / (0.012 * c) < 2.0e-2

    def test_gravitational_correction(self):
        accel_no_grav = self.sail.acceleration(2 * solar_radius, self.payload_mass, include_gravity=False)
        accel_with_grav = self.sail.acceleration(2 * solar_radius, self.payload_mass, include_gravity=True)
        assert accel_with_grav / (m / s ** 2) < accel_no_grav / (m / s ** 2)

        v_fin_no_grav = self.sail.final_velocity(self.payload_mass, 2 * solar_radius, include_gravity=False)
        v_fin_with_grav = self.sail.final_velocity(self.payload_mass, 2 * solar_radius, include_gravity=True)
        assert v_fin_with_grav / (m / s) < v_fin_no_grav / (m / s)

    def test_destination_stellar_luminosity(self):
        proxima_luminosity = 6.5e23 * watt
        accel_sun = self.sail.acceleration(2 * solar_radius, self.payload_mass)
        accel_proxima = self.sail.acceleration(2 * solar_radius,
                                               self.payload_mass,
                                               stellar_luminosity=proxima_luminosity)
        assert accel_proxima / (m / s ** 2) < accel_sun / (m / s ** 2)
        assert abs((accel_proxima / accel_sun) - (6.5e23 / 3.83e26)) < 1.0e-4

    def test_keyword_only_arguments_enforcement(self):
        """Test that methods require keyword arguments for optional configuration parameters"""
        # SolarSail.__init__ requires kwargs after sail_mass, sail_radius
        with pytest.raises(TypeError):
            SolarSail(self.sail_mass, self.sail_radius, 0.95)  # pylint: disable=too-many-positional-arguments

        # SolarSail.acceleration requires kwargs after relative_position_from_star, payload_mass
        with pytest.raises(TypeError):
            self.sail.acceleration(2 * solar_radius, self.payload_mass, 0.01 * g)  # pylint: disable=too-many-positional-arguments

        # SolarSail.final_velocity requires kwargs after payload_mass, initial_distance_from_star
        with pytest.raises(TypeError):
            self.sail.final_velocity(self.payload_mass, 2 * solar_radius, 1.0 * kg)  # pylint: disable=too-many-positional-arguments





