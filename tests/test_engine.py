"""Tests for the Engine class"""
from scimath.units.length import kilometers as m, km
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

    def test_set_target_delta_v(self, ):
        """Test the set_target_delta_v method"""
        target_delta_v = 1000 * m / s
        initial_fuel_mass = self.engine.fuel_mass
        self.engine.set_target_delta_v(target_delta_v, self.payload_mass)
        assert self.engine.fuel_mass / kg < initial_fuel_mass / kg