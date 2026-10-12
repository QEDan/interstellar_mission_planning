"""Tests for SWIMMER engine plasma physics, ISM environments, drag, and braking."""

import numpy as np
import pytest
from scimath.units.length import meters as m, kilometers as km, astronomical_unit as au, light_year as ly
from scimath.units.mass import kilograms as kg
from scimath.units.time import seconds as s, years as yr
from scimath.units.temperature import kelvin as K
from scimath.units.SI import watt

from src.swimmer import Swimmer, ISMEnvironment, TripProfile
from src.constants import c


class TestSwimmerPhysics:
    """Test suite for SWIMMER plasma physics, galactic environments, drag, and braking."""

    def setup_method(self):
        """Set up standard test fixtures."""
        self.pusher_area = 2.0e19 * (m ** 2)
        self.total_mass = 1.0e6 * kg
        self.swimmer = Swimmer(self.pusher_area)

    def test_initialization_defaults(self):
        """Test default Swimmer attributes preserve baseline physics values."""
        assert self.swimmer.pusher_area / (m ** 2) == pytest.approx(2.0e19)
        assert self.swimmer.pusher_mass() / kg == pytest.approx(2.0e19)
        assert self.swimmer.ion_density / ((m / 100) ** -3) == pytest.approx(0.07)
        assert self.swimmer.neutral_density / ((m / 100) ** -3) == pytest.approx(0.10)

        # Shed area
        delta_area = 1.0e18 * (m ** 2)
        self.swimmer.shed_area(delta_area)
        assert self.swimmer.pusher_area / (m ** 2) == pytest.approx(1.9e19)
        assert self.swimmer.pusher_mass() / kg == pytest.approx(1.9e19)

    def test_ism_environments_density_spectrum(self):
        """Test ISM environments span orders of magnitude from 10^-3 to 10^4 cm^-3."""
        him = ISMEnvironment.hot_ionized()
        lic = ISMEnvironment.local_cloud()
        wim = ISMEnvironment.warm_ionized()
        wnm = ISMEnvironment.warm_neutral()
        cnm = ISMEnvironment.cold_neutral()
        mc_diffuse = ISMEnvironment.molecular_cloud(density_cm3=100.0)
        mc_dense = ISMEnvironment.molecular_cloud(density_cm3=10000.0)

        # Hot ionized medium: 10^-3 cm^-3 regime
        assert him.ion_density / ((m / 100) ** -3) == pytest.approx(0.005)
        assert him.temperature / K == pytest.approx(1.0e6)
        assert him.debye_length() / m > 500.0  # ~975 m Debye length

        # Local Interstellar Cloud: ~0.07 cm^-3 regime
        assert lic.ion_density / ((m / 100) ** -3) == pytest.approx(0.07)
        assert lic.debye_length() / m == pytest.approx(21.82, rel=0.05)

        # Warm Ionized Medium: ~0.3 cm^-3 regime
        assert wim.ion_density / ((m / 100) ** -3) == pytest.approx(0.30)

        # Warm Neutral Medium: ~0.5 cm^-3 total
        assert wnm.total_number_density / ((m / 100) ** -3) == pytest.approx(0.53)

        # Cold Neutral Medium: ~30 cm^-3 regime
        assert cnm.total_number_density / ((m / 100) ** -3) == pytest.approx(30.04)
        assert cnm.debye_length() / m < 5.0  # strongly shielded

        # Molecular Cloud: 10^4 cm^-3 regime
        assert mc_dense.neutral_density / ((m / 100) ** -3) == pytest.approx(10000.0)
        assert mc_diffuse.neutral_density / ((m / 100) ** -3) == pytest.approx(100.0)

        # Verify density spans over 6 orders of magnitude
        min_density = float(him.ion_density / ((m / 100) ** -3))
        max_density = float(mc_dense.total_number_density / ((m / 100) ** -3))
        assert max_density / min_density > 1.0e6

    def test_heliosphere_environment(self):
        """Test heliospheric stellar wind density profile decays as 1/r^2."""
        env_1au = ISMEnvironment.heliosphere(distance_from_star=1.0 * au)
        env_10au = ISMEnvironment.heliosphere(distance_from_star=10.0 * au)
        env_100au = ISMEnvironment.heliosphere(distance_from_star=100.0 * au)

        # Density at 1 AU ~ 7.07 cm^-3
        assert env_1au.ion_density / ((m / 100) ** -3) == pytest.approx(7.07, rel=0.01)
        # Density at 10 AU ~ 0.14 cm^-3
        assert env_10au.ion_density / ((m / 100) ** -3) == pytest.approx(0.14, rel=0.05)
        # Density at 100 AU ~ 0.07 cm^-3 (transitions to LIC)
        assert env_100au.ion_density / ((m / 100) ** -3) == pytest.approx(0.07, rel=0.05)

        # Streaming velocity is 450 km/s outward
        assert env_1au.streaming_velocity / (km / s) == pytest.approx(450.0)

    def test_trip_profile_sol_to_alpha_centauri(self):
        """Test full trip profile modeling Sol heliosphere, ISM, and destination astrosphere."""
        dest_dist = 4.244 * ly
        trip = TripProfile(destination_distance=dest_dist)

        # Near Sol (10 AU)
        env_sol = trip.environment_at(10.0 * au)
        assert not trip.is_interstellar(10.0 * au)
        assert env_sol.streaming_velocity / (km / s) == pytest.approx(450.0)
        assert float(env_sol.ion_density / ((m / 100) ** -3)) > 0.10

        # Mid-cruise (1 light year into voyage)
        env_ism = trip.environment_at(1.0 * ly)
        assert trip.is_interstellar(1.0 * ly)
        assert env_ism.streaming_velocity / (km / s) == pytest.approx(0.0)
        assert env_ism.ion_density / ((m / 100) ** -3) == pytest.approx(0.07)

        # Approaching destination star (10 AU from Alpha Centauri)
        pos_dest = dest_dist - 10.0 * au
        env_dest = trip.environment_at(pos_dest)
        assert not trip.is_interstellar(pos_dest)
        # Headwind streaming velocity towards incoming ship (-450 km/s)
        assert env_dest.streaming_velocity / (km / s) == pytest.approx(-450.0)

    def test_parasitic_drag_modeling(self):
        """Test drag modeling accounts for neutral gas and hull collisions."""
        v = 0.01 * c
        drag_lic = self.swimmer.drag_force(v, environment=ISMEnvironment.local_cloud())
        drag_mc = self.swimmer.drag_force(
            v, environment=ISMEnvironment.molecular_cloud(density_cm3=1000.0)
        )

        assert drag_lic / (kg * m / (s ** 2)) > 0.0
        # Drag in a molecular cloud (1000 cm^-3) must be orders of magnitude higher than in LIC
        assert drag_mc / drag_lic > 1.0e3

        # Drag at zero relative velocity is zero
        assert self.swimmer.drag_force(0.0 * m / s) / (kg * m / (s ** 2)) == pytest.approx(0.0)

    def test_unpowered_coasting_drag_mitigates_classical_bug(self):
        """Test unpowered coasting (P=0, braking=False) experiences non-zero decelerating drag."""
        v = 0.01 * c
        # When unpowered (P = 0), net force must oppose motion (negative)
        net_f = self.swimmer.net_force(0.0 * watt, v, braking=False)
        assert net_f / (kg * m / (s ** 2)) < 0.0

        # Acceleration returns non-zero deceleration magnitude
        accel = self.swimmer.acceleration(0.0 * watt, v, self.total_mass, braking=False)
        assert accel / (m / (s ** 2)) > 0.0

    def test_active_swimmer_thrust_mitigates_drag(self):
        """Test that active SWIMMER power converts incoming ion momentum into gross thrust."""
        v = 0.01 * c
        power = 1.0e13 * watt

        net_f = self.swimmer.net_force(power, v, braking=False)
        # Gross thrust exceeds ram drag, providing positive net acceleration
        assert net_f / (kg * m / (s ** 2)) > 0.0

        accel = self.swimmer.acceleration(power, v, self.total_mass, braking=False)
        assert accel / (m / (s ** 2)) > 0.0

    def test_braking_regimes_passive_and_powered(self):
        """Test that braking is validated across passive (P=0) and powered (P>0) regimes."""
        v = 0.01 * c

        # Passive braking (P = 0, electric sail / magsail reflection limit)
        f_passive_brake = self.swimmer.net_force(0.0 * watt, v, braking=True)
        assert f_passive_brake / (kg * m / (s ** 2)) > 0.0

        # Powered braking (P > 0) actively accelerates ions forward
        f_powered_brake = self.swimmer.net_force(1.0e13 * watt, v, braking=True)
        assert f_powered_brake / (kg * m / (s ** 2)) > f_passive_brake / (kg * m / (s ** 2))

        # Check smooth limit: as power approaches zero, powered braking approaches passive braking
        f_near_zero_power = self.swimmer.net_force(1.0e-3 * watt, v, braking=True)
        assert float(f_near_zero_power / (kg * m / (s ** 2))) == pytest.approx(
            float(f_passive_brake / (kg * m / (s ** 2))), rel=1.0e-4
        )

    def test_stopping_potential_physics(self):
        """Test stopping potential calculation for reflecting ions."""
        v1 = 1000.0 * km / s
        v2 = 0.01 * c

        # Stopping potential phi_stop = m_p * v^2 / (2 * e)
        # At 1000 km/s (~5.2 keV), voltage ~ 5.2 kV
        volts1 = self.swimmer.stopping_potential(v1)
        # Extract numerical value in volts
        v_val1 = float(volts1 / ((kg * (m ** 2)) / ((s ** 3) * (watt / (kg * (m ** 2) / (s ** 3))))))
        assert v_val1 == pytest.approx(5217.4, rel=0.01)

        # At 0.01 c (~47 keV), voltage ~ 47 kV
        volts2 = self.swimmer.stopping_potential(v2)
        v_val2 = float(volts2 / ((kg * (m ** 2)) / ((s ** 3) * (watt / (kg * (m ** 2) / (s ** 3))))))
        assert v_val2 == pytest.approx(46950.0, rel=0.01)
        assert v_val2 > v_val1

    def test_low_velocity_smooth_transition_with_thermal_speed(self):
        """Test low velocity limit smoothly transitions using plasma thermal velocity."""
        # Near zero velocity
        v_near_zero = 1.0 * m / s
        env = ISMEnvironment.local_cloud()

        # Thermal velocity in 7000 K plasma is ~12 km/s
        v_th = env.thermal_velocity()
        assert v_th / (km / s) == pytest.approx(12.15, rel=0.05)

        # Net force at near-zero velocity does not suffer from zero-division or discontinuity
        accel = self.swimmer.acceleration(1.0e12 * watt, v_near_zero, self.total_mass, braking=False)
        assert accel / (m / (s ** 2)) > 0.0
