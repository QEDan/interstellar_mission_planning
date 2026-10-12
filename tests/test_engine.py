"""Tests for the Engine class"""
from scimath.units.length import meters as m, kilometers as km
from scimath.units.mass import kilograms as kg
from scimath.units.time import seconds as s

from src import Engine


class TestEngine:
    """Test Engine Class"""
    def setup_method(self):
        self.fuel_mass = 100000 * kg
        self.payload_mass = 50 * kg
        self.engine = Engine(self.fuel_mass)

    def test_init(self):
        """Test class initialization"""
        assert isinstance(self.engine, Engine)
        assert self.engine.fuel_mass == self.fuel_mass

    def test_burn_fuel(self):
        """Test the burn_fuel method"""
        burnt_fuel_mass = 0.5 * self.fuel_mass
        delta_v = self.engine.burn_fuel(burnt_fuel_mass, self.payload_mass)
        assert isinstance(delta_v / (m / s), float)

    def test_set_target_delta_v(self):
        """Test the set_target_delta_v method"""
        target_delta_v = 1000 * m / s
        initial_fuel_mass = self.engine.fuel_mass
        self.engine.set_target_delta_v(target_delta_v, self.payload_mass)
        assert self.engine.fuel_mass / kg < initial_fuel_mass / kg

    def test_relativistic_invertibility(self):
        """Test set_target_delta_v accurately burns fuel required for target_delta_v"""
        target_dv = 1000000.0 * (m / s)
        engine = Engine(1.0e10 * kg, exhaust_velocity=3000 * km / s)
        initial_fuel = engine.fuel_mass
        engine.set_target_delta_v(target_dv, 500.0 * kg)

        burnt_fuel = initial_fuel - engine.fuel_mass
        verify_engine = Engine(initial_fuel, exhaust_velocity=3000 * km / s)
        actual_dv = verify_engine.burn_fuel(burnt_fuel, 500.0 * kg)

        assert abs((actual_dv - target_dv) / target_dv) < 1.0e-5