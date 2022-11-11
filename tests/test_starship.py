"""Tests for Starship class"""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scimath.units.length import kilometers as km, astronomical_unit
from scimath.units.length import light_year as ly
from scimath.units.length import meters as m
from scimath.units.mass import kilograms as kg
from scimath.units.time import seconds as s, years

from src import SolarSail, Engine, Starship, Swimmer
from src.constants import c, year


class TestStarship:
    """Tests for the Starship class"""
    def setup_method(self):
        self.fuel_mass = 1.0e10 * kg
        self.engine = Engine(self.fuel_mass)
        self.payload_mass = 50.0 * kg
        self.starship = Starship(self.payload_mass, {'main': self.engine})

    def test_init(self):
        """Test starship initialization"""
        assert isinstance(self.starship, Starship)

    def test_log_entry(self):
        """Test that we can log entries"""
        self.starship.log_entry('test1')
        self.starship.log_entry('test2')
        assert len(self.starship.log_messages) == 3
        assert len(self.starship.history) == 3
        assert self.starship.log_messages[1] == 'test1'
        assert isinstance(self.starship.history[0], dict)

    def test_total_mass(self):
        """Test total mass is correct"""
        mass = self.starship.total_mass()
        assert isinstance(mass / kg, float)
        assert mass == self.starship.payload_mass + self.fuel_mass

    def test_fuel_mass(self):
        """Test fuel mass is correct"""
        mass = self.starship.fuel_mass()
        assert isinstance(mass / kg, float)
        assert mass == self.fuel_mass

    @pytest.mark.parametrize('generate_electricity', [True, False])
    def test_generate_electricity(self, generate_electricity):
        delta_t = 1 * years
        total_energy = self.starship.electrical_power * delta_t
        initial_payload_mass = self.starship.payload_mass
        initial_fuel_mass = self.starship.fuel_mass()
        self.starship.generate_electricity(total_energy)
        mass_defect = (initial_fuel_mass + initial_payload_mass) - \
                      (self.starship.payload_mass + self.starship.fuel_mass())
        assert self.starship.payload_mass / kg > initial_payload_mass / kg
        assert self.starship.fuel_mass() / kg < initial_fuel_mass / kg
        assert mass_defect / kg > 0.0


    @pytest.mark.parametrize('fuel_fraction', [0.1, 0.5, 0.9])
    @pytest.mark.parametrize('direction', [1, -1])
    def test_accelerate_fuel_mass(self, fuel_fraction, direction):
        """Test accelerate using specific fuel mass values"""
        velocity = self.starship.accelerate(
            fuel_mass=fuel_fraction * self.fuel_mass,
            direction=direction)
        assert isinstance(velocity / (m / s), float)
        assert abs(velocity / c) < 0.5
        assert abs(self.starship.fuel_mass() -
                   (1 - fuel_fraction) * self.fuel_mass) / self.fuel_mass < 0.01
        assert self.starship.time / s > 0
        if direction == 1:
            assert self.starship.position / m > 0
        else:
            assert self.starship.position / m < 0

    @pytest.mark.parametrize('target_velocity', [0.001 * c, 0.01 * c, 0.02 * c])
    @pytest.mark.parametrize('direction', [1, -1])
    def test_accelerate_velocity(self, target_velocity, direction):
        """Test accelerate using specific target velocities"""
        velocity = self.starship.accelerate(
            target_velocity=direction * target_velocity,
            direction=direction)
        assert isinstance(velocity / (m / s), float)
        assert abs(velocity - direction * target_velocity) / target_velocity < 1.0e-3
        assert self.starship.fuel_mass() / kg < self.fuel_mass / kg
        assert self.starship.time / s > 0
        if direction == 1:
            assert self.starship.position / m > 0
        else:
            assert self.starship.position / m < 0

    @pytest.mark.parametrize('fuel_fraction', [0.1, 0.5, 0.9])
    @pytest.mark.parametrize('direction', [1, -1])
    def test_decelerate_fuel_mass(self, fuel_fraction, direction):
        """Test deceleration using specific fuel masses"""
        initial_velocity = 0.1 * c
        self.starship.velocity = initial_velocity
        velocity = self.starship.accelerate(fuel_mass=fuel_fraction * self.fuel_mass,
                                            direction=direction)
        assert isinstance(velocity / (m / s), float)
        if direction == 1:
            assert velocity / c > initial_velocity / c
        else:
            assert velocity / c < initial_velocity / c
        assert abs(self.starship.fuel_mass() -
                   (1 - fuel_fraction) * self.fuel_mass) / self.fuel_mass < 0.01
        assert self.starship.time / s > 0

    @pytest.mark.parametrize('target_delta_velocity',
                             [0.001 * c,
                              0.01 * c,
                              0.02 * c])
    @pytest.mark.parametrize('direction', [1, -1])
    def test_decelerate_velocity(self, target_delta_velocity, direction):
        """Test decelerate using specific target velocities"""
        self.starship.engines['main'].fuel_mass = self.fuel_mass
        initial_velocity = 0.03 * c
        target_velocity = initial_velocity + direction * target_delta_velocity
        self.starship.velocity = initial_velocity
        velocity = self.starship.accelerate(target_velocity=target_velocity,
                                            direction=direction)
        assert isinstance(velocity / (m / s), float)
        assert abs(velocity - target_velocity) / target_velocity < 1.0e-3
        assert self.starship.fuel_mass() / kg < self.fuel_mass / kg
        assert self.starship.time / s > 0
        assert self.starship.position / m > 0

    def test_not_enough_fuel(self):
        """Test that we get an exception when insufficient fuel"""
        self.starship.engines['main'].fuel_mass = 1.0e3 * kg
        with pytest.raises(ValueError):
            _ = self.starship.accelerate(target_velocity=0.1 * c)

    @pytest.mark.parametrize('direction', [1, -1])
    def test_cruise(self, direction):
        """Test cruising"""
        distance = 1000 * astronomical_unit
        self.starship.velocity = direction * 0.1 * c
        self.starship.cruise(distance)
        assert abs(direction * self.starship.position - distance) / distance < 1.0e-3
        assert abs(direction * self.starship.velocity /
                   (distance / self.starship.time) - 1) < 1.0e-3
        assert len(self.starship.history) == 2

    @pytest.mark.parametrize('direction', [1, -1])
    def test_wait(self, direction):
        """Test waiting"""
        time = 1.0e3 * s
        self.starship.velocity = direction * 0.1 * c
        self.starship.wait(time)
        assert self.starship.time == time
        assert abs(self.starship.position / m - self.starship.velocity * time / m) < 1.0e-3
        assert len(self.starship.history) == 2

    def test_parse_logs(self):
        """Test parsing of ship logs"""
        time = 1.0e3 * s
        n_logs = 10
        self.starship.velocity = 0.1 * c
        for _ in range(n_logs - 1):
            self.starship.wait(time)
        positions, velocities, fuels, times = self.starship.parse_logs()
        for l in [positions, velocities, fuels, times]:
            assert len(l) == n_logs
        for i in range(n_logs):
            # Units are assumed for parse_logs
            assert isinstance(positions[i], float)
            assert isinstance(velocities[i], float)
            assert isinstance(fuels[i], float)
            assert isinstance(times[i], float)

    def test_plot_history(self):
        """Test that plot_history returns a figure"""
        fig = self.starship.plot_history()
        assert isinstance(fig, plt.Figure)
        plt.close()

    @pytest.mark.parametrize('direction', [1, -1])
    def test_sail_out(self, direction):
        sail_area_density = 0.00003 * kg / m ** 2  # Carbon nanotube sheets
        sail_radius = 6000 * km
        sail_mass = sail_radius ** 2 * np.pi * sail_area_density
        solar_sail = SolarSail(sail_mass, sail_radius, reflectivity=0.98)
        initial_distance = 0.02 * astronomical_unit
        self.starship.solar_sail = solar_sail
        self.starship.position = direction * initial_distance
        self.starship.sail(None,
                           position_of_star=0.0 * m)
        expected_velocity = direction * 0.9 * self.starship.solar_sail.final_velocity(
            self.starship.total_mass() - solar_sail.sail_mass,
            initial_distance,
        )
        assert abs((self.starship.velocity - expected_velocity)
                   / expected_velocity) < 0.1
        assert direction * self.starship.position / astronomical_unit > 1.0e-2

    @pytest.mark.parametrize('direction', [1, -1])
    def test_sail_decelerate(self, direction):
        sail_area_density = 0.00003 * kg / m ** 2  # Carbon nanotube sheets
        sail_radius = 6000 * km
        sail_mass = sail_radius ** 2 * np.pi * sail_area_density
        solar_sail = SolarSail(sail_mass, sail_radius, reflectivity=0.98)
        initial_distance = direction * -8.0 * astronomical_unit
        self.starship.position = initial_distance
        self.starship.solar_sail = solar_sail
        initial_velocity = 0.9 * self.starship.solar_sail.final_velocity(
            self.starship.total_mass() - solar_sail.sail_mass,
            initial_distance,
        ) * direction
        self.starship.velocity = initial_velocity
        self.starship.sail(None,
                           position_of_star=0.0 * m,
                           max_sail_time=100.00 * 3600 * 24 * 365 * s)
        assert abs(self.starship.velocity / initial_velocity) < 1.0e-3

    @pytest.mark.parametrize('direction', [1, -1])
    def test_sail_decelerate_destination(self, direction):
        sail_area_density = 0.00003 * kg / m ** 2  # Carbon nanotube sheets
        sail_radius = 6000 * km
        sail_mass = sail_radius ** 2 * np.pi * sail_area_density
        solar_sail = SolarSail(sail_mass, sail_radius, reflectivity=0.98)
        total_distance = 4.244 * ly
        initial_distance = total_distance + direction * -8.0 * astronomical_unit
        self.starship.position = initial_distance
        self.starship.solar_sail = solar_sail
        initial_velocity = 0.9 * self.starship.solar_sail.final_velocity(
            self.starship.total_mass() - solar_sail.sail_mass,
            initial_distance,
        ) * direction
        self.starship.velocity = initial_velocity
        self.starship.sail(0.0 * c,
                           position_of_star=total_distance)
        assert abs(self.starship.velocity) / initial_velocity < 1.0e-3


    @pytest.mark.parametrize('direction', [1, -1])
    def test_sail_accel_deccel(self, direction):
        sail_area_density = 0.00003 * kg / m ** 2  # Carbon nanotube sheets
        sail_radius = 6000 * km
        sail_mass = sail_radius ** 2 * np.pi * sail_area_density
        solar_sail = SolarSail(sail_mass, sail_radius, reflectivity=0.98)
        initial_distance = 0.02 * astronomical_unit
        self.starship.position = direction * initial_distance
        self.starship.solar_sail = solar_sail
        self.starship.sail(None,
                           position_of_star=0.0 * m)
        self.starship.position *= -1.0
        self.starship.sail(None,
                           position_of_star=0.0 * m)
        assert abs(self.starship.velocity / c) < 1.0e-5
        assert direction * self.starship.position / m < 1.0e3
        assert int(np.sign(self.starship.position / m)) == -1 * direction

    @pytest.mark.parametrize('velocity_direction', [1, -1])
    @pytest.mark.parametrize('accel_direction', [1, -1])
    def test_swim(self, velocity_direction, accel_direction):
        power_delivered = 1.0e13 * kg * m ** 2 / s ** 3
        pusher_area = 2.0e19 * m ** 2
        swim_time = 1.0 * year
        swimmer = Swimmer(pusher_area)
        initial_velocity = velocity_direction * 0.01 * c
        self.starship.velocity = initial_velocity
        self.starship.swimmer = swimmer
        self.starship.swim(
            power_delivered,
            swim_time,
            direction=accel_direction
        )

        assert np.sign(self.starship.velocity / (m / s)
                       - initial_velocity / (m / s)) == accel_direction
