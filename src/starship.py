"""Starship class"""

from typing import Optional, Tuple, Dict, Any, Union, Sequence
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

from scimath.units.unit import unit
from scimath.units.SI import watt
from scimath.units.energy import MeV
from scimath.units.length import kilometers as km
from scimath.units.length import astronomical_unit
from scimath.units.length import light_year as ly
from scimath.units.length import meters as m
from scimath.units.mass import kilograms as kg
from scimath.units.time import seconds as s
from scimath.units.time import years as yr

from src.constants import c, g, solar_mass, mass_3He, mass_2H
from src.solar_sail import SolarSail
from src.swimmer import Swimmer
from src.utils import v_escape


class NumericalIntegrationError(RuntimeError):
    """Exception raised when numerical integration fails to solve or converge."""


class Starship:
    """A Starship that uses engines to accelerate."""
    def __init__(self,
                 payload_mass,
                 engines: dict,
                 *,
                 initial_velocity: unit = 0 * m / s,
                 initial_position: unit = 0 * km,
                 initial_time: unit = 0 * s,
                 destination_distance: unit = 4.244 * ly,
                 electrical_power: unit = 1.5e11 * watt,
                 solar_sail: SolarSail = None,
                 swimmer: Swimmer = None,
                 source_star_mass: unit = solar_mass,
                 destination_star_mass: unit = solar_mass,
                 source_stellar_luminosity: unit = 3.83e26 * watt,
                 destination_stellar_luminosity: unit = 3.83e26 * watt,
                 departure_distance: unit = 1 * astronomical_unit
                 ):
        self.payload_mass = payload_mass
        self.engines = engines
        self.velocity = initial_velocity
        self.position = initial_position
        self.time = initial_time
        self.destination_distance = destination_distance
        self.electrical_power = electrical_power
        self.solar_sail = solar_sail
        self.swimmer = swimmer
        self.source_star_mass = source_star_mass
        self.destination_star_mass = destination_star_mass
        self.source_stellar_luminosity = source_stellar_luminosity
        self.destination_stellar_luminosity = destination_stellar_luminosity
        self.departure_distance = departure_distance
        self.history = []
        self.log_messages = []
        self.log_entry()

    def log_entry(self, message: str = ''):
        """Enter current data into the log"""

        new_log = {'time': self.time,
                   'position': self.position,
                   'velocity': self.velocity,
                   'fuel_mass': self.fuel_mass()}
        self.history.append(new_log)
        self.log_messages.append(message)

    def total_mass(self):
        """Return the payload plus fuel masses"""
        total_mass = self.payload_mass + self.fuel_mass()
        if self.solar_sail is not None:
            total_mass += self.solar_sail.sail_mass
        if self.swimmer is not None:
            total_mass += self.swimmer.pusher_mass()
        return total_mass

    def fuel_mass(self):
        """Return the current fuel mass"""
        mass = sum(e.fuel_mass / kg for e in self.engines.values()) * kg
        return mass

    def source_escape_velocity(self, departure_distance: unit = None) -> unit:
        """Return the escape velocity from the source star."""
        if departure_distance is None:
            departure_distance = self.departure_distance
        return v_escape(departure_distance, self.source_star_mass)

    def destination_escape_velocity(self, distance_from_star: unit = None) -> unit:
        """Return the escape velocity at the destination star."""
        if distance_from_star is None:
            distance_from_star = self.departure_distance
        return v_escape(distance_from_star, self.destination_star_mass)

    def generate_electricity(self,
                             energy_needed: unit,
                             efficiency: float = 0.7) -> None:
        """Assume the engines will be used for electricity generation rather than propulsion
        to adjust masses.

        This calculation assumes the main reaction is 3He + 2H -> p + 4He.
        Fuel is consumed from engine tanks to generate the required electrical energy accounting
        for conversion efficiency. Products and waste energy are exhausted into space.
        """
        energy_fusion = 18.354 * MeV
        num_reactions = energy_needed / (efficiency * energy_fusion)
        fuel_mass_lost = num_reactions * (mass_3He + mass_2H)
        if self.fuel_mass() / kg < fuel_mass_lost / kg:
            raise ValueError(f"Cannot generate {energy_needed} of electricity with "
                             f"only {self.fuel_mass()} of fuel.")
        total_fuel_kg = self.fuel_mass() / kg
        if total_fuel_kg > 0:
            for engine in self.engines.values():
                fraction = (engine.fuel_mass / kg) / total_fuel_kg
                engine.fuel_mass -= fraction * fuel_mass_lost

    def _burn_acceleration(self,
                           engine_name: str,
                           target_velocity: unit,
                           *,
                           include_gravity: bool,
                           departure_distance: Optional[unit]) -> unit:
        """Execute engine burn for acceleration to a target velocity."""
        v_curr_ms = self.velocity / (m / s)
        target_v_ms = target_velocity / (m / s)
        v_esc = self.source_escape_velocity(departure_distance)
        v_esc_ms = v_esc / (m / s)
        ship_dry_mass = self.total_mass() - self.engines[engine_name].fuel_mass

        v_target_mag = abs(target_v_ms)
        v_curr_mag = abs(v_curr_ms)
        if include_gravity:
            v_req_mag = np.sqrt(v_target_mag ** 2 + v_esc_ms ** 2)
            delta_v_burn_ms = max(0.0, v_req_mag - v_curr_mag)
        else:
            delta_v_burn_ms = abs(v_target_mag - v_curr_mag)
        delta_v_burn = delta_v_burn_ms * (m / s)
        self.engines[engine_name].set_target_delta_v(delta_v_burn, ship_dry_mass)
        self.velocity = target_velocity
        return delta_v_burn

    def _burn_deceleration(self,
                           engine_name: str,
                           target_velocity: unit,
                           *,
                           include_gravity: bool,
                           destination_distance_from_star: Optional[unit]) -> unit:
        """Execute engine burn for deceleration to a target velocity."""
        v_curr_ms = self.velocity / (m / s)
        target_v_ms = target_velocity / (m / s)
        v_esc = self.destination_escape_velocity(destination_distance_from_star)
        v_esc_ms = v_esc / (m / s)
        ship_dry_mass = self.total_mass() - self.engines[engine_name].fuel_mass

        v_target_mag = abs(target_v_ms)
        v_curr_mag = abs(v_curr_ms)
        if include_gravity:
            v_inbound_ms = np.sqrt(v_curr_mag ** 2 + v_esc_ms ** 2)
            delta_v_burn_ms = max(0.0, v_inbound_ms - v_target_mag)
        else:
            delta_v_burn_ms = abs(v_curr_mag - v_target_mag)
        delta_v_burn = delta_v_burn_ms * (m / s)
        self.engines[engine_name].set_target_delta_v(delta_v_burn, ship_dry_mass)
        self.velocity = target_velocity
        return delta_v_burn

    def accelerate(self,
                   engine_name: str = 'main',
                   target_velocity: unit = 0 * km / s,
                   *,
                   fuel_mass: unit = None,
                   direction: int = 1,
                   acceleration: unit = g,
                   generate_electricity: bool = True,
                   include_gravity: bool = True,
                   departure_distance: unit = None,
                   destination_distance_from_star: unit = None) -> unit:
        """Accelerate the ship by burning a specified quantity of fuel.

        If no fuel mass is specified, a target_velocity should be specified instead.

        Args:
            engine_name: src
                Name of the engine
            target_velocity: unit (speed)
                Target velocity to accelerate to.
            fuel_mass: unit (mass)
                Amount of fuel mass to burn
            direction: int
                If 1, acceleration is in the positive direction
                If -1, acceleration is in the negative direction
            acceleration: unit (acceleration)
                Engine burn acceleration rate
            generate_electricity: bool
                Whether to generate electricity during maneuver
            include_gravity: bool
                Whether target-velocity burns include gravitational escape
                velocity corrections. Ignored for fuel-mass burns, which apply
                rocket Δv along `direction`.
            departure_distance: unit (distance)
                Distance from source star for escape velocity calculation
            destination_distance_from_star: unit (distance)
                Distance from destination star for escape velocity calculation

        Returns:
            unit (speed)
                New velocity of the starship
        """
        v_curr_ms = self.velocity / (m / s)
        target_v_ms = target_velocity / (m / s)

        initial_v_ms = v_curr_ms
        if fuel_mass is not None:
            # Fuel-limited burns apply rocket Δv along `direction`. Do not treat
            # leftover speed as already pointing in the thrust direction, and do
            # not fold stellar escape speed into a mid-course velocity jump.
            ship_dry_mass = self.total_mass() - self.engines[engine_name].fuel_mass
            delta_v = self.engines[engine_name].burn_fuel(fuel_mass, ship_dry_mass)
            self.velocity += direction * delta_v
        else:
            is_decelerating = abs(target_v_ms) < abs(v_curr_ms)
            if not is_decelerating:
                delta_v = self._burn_acceleration(
                    engine_name, target_velocity,
                    include_gravity=include_gravity,
                    departure_distance=departure_distance)
            else:
                delta_v = self._burn_deceleration(
                    engine_name, target_velocity,
                    include_gravity=include_gravity,
                    destination_distance_from_star=destination_distance_from_star)

        final_v_ms = self.velocity / (m / s)
        accel_ms2 = acceleration / (m / s ** 2)
        delta_v_ms = delta_v / (m / s)
        delta_t_s = abs(delta_v_ms) / accel_ms2
        delta_t = delta_t_s * s
        self.time += delta_t
        delta_pos_m = 0.5 * (initial_v_ms + final_v_ms) * delta_t_s
        self.position += delta_pos_m * m
        if generate_electricity:
            self.generate_electricity(self.electrical_power * delta_t)

        if abs(self.velocity / c) > 0.5:
            raise NotImplementedError(
                "This ship is travelling at relativistic speeds. This is not currently supported.")

        self.log_entry(
            f"year {(self.time - delta_t) / yr:0.1f} - "
            f"Acceleration: {acceleration / g:0.4f} g for {delta_t / yr:0.2e} years. "
            f" New velocity is {self.velocity / c:0.2e} c. "
            f" {self.fuel_mass() / kg:0.2e} kg of fuel remaining."
        )

        return self.velocity

    def cruise(self, distance: unit, generate_electricity: bool = True):
        """Travel at current velocity without acceleration"""
        if self.velocity == 0:
            raise ValueError("The starship is not moving. Can't cruise.")
        delta_pos = distance * np.sign(self.velocity / (m / s))
        delta_t = np.abs(distance / self.velocity)
        self.position += delta_pos
        self.time += delta_t
        if generate_electricity:
            self.generate_electricity(self.electrical_power * delta_t)
        self.log_entry(
            f"year {(self.time - delta_t) / yr:0.1f} - "
            f"Cruise: {delta_t / yr:0.2e} years to complete. "
            f"Distance={distance / ly:0.2e} lightyears")

    def wait(self, time: unit, generate_electricity: bool = True):
        """Pass time with no acceleration"""
        self.time += time
        distance = self.velocity * time
        self.position += distance
        if generate_electricity:
            self.generate_electricity(self.electrical_power * time)
        self.log_entry(
            f"year {(self.time - time) / yr:0.1f} - Waited: {time / yr:0.2e} years. "
            f"Distance={distance / ly:0.2e} lightyears")

    def _resolve_star_properties(self,
                                 position_of_star: unit,
                                 star_mass: Optional[unit],
                                 stellar_luminosity: Optional[unit]) -> Tuple[unit, unit]:
        """Resolve mass and luminosity for source or destination star."""
        is_destination = abs((position_of_star - self.destination_distance) / m) < 1e3
        if star_mass is None:
            resolved_mass = self.destination_star_mass if is_destination else self.source_star_mass
        else:
            resolved_mass = star_mass

        if stellar_luminosity is None:
            resolved_luminosity = self.destination_stellar_luminosity if is_destination \
                else self.source_stellar_luminosity
        else:
            resolved_luminosity = stellar_luminosity

        return resolved_mass, resolved_luminosity

    def _integrate_sail(self,
                        relative_position_to_star: unit,
                        *,
                        max_sail_time: unit,
                        max_accel: Optional[unit],
                        star_mass: unit,
                        stellar_luminosity: unit,
                        include_gravity: bool,
                        rtol: float = 1e-6,
                        atol: Union[float, Sequence[float]] = 1e-9,
                        method: str = 'RK45',
                        max_step: float = np.inf,
                        solver_options: Optional[Dict[str, Any]] = None):
        """Solve ODE trajectory for solar sailing with robust error handling."""
        max_sail_time_s = max_sail_time / s
        if max_sail_time_s <= 0:
            raise ValueError(f"max_sail_time must be positive, got {max_sail_time_s} s.")

        rel_pos_m = relative_position_to_star / m
        if not np.isfinite(rel_pos_m):
            raise ValueError("relative_position_to_star must be a finite value.")
        if rel_pos_m == 0:
            raise ValueError(
                "Initial relative_position_to_star cannot be zero (singularity at star center)."
            )

        init_v_ms = self.velocity / (m / s)
        if not np.isfinite(init_v_ms):
            raise ValueError("Initial velocity must be a finite value.")

        def integrand(_, pos_vel):
            if not np.all(np.isfinite(pos_vel)):
                raise FloatingPointError(
                    f"Non-finite state encountered in sailing integrand: {pos_vel}"
                )
            pos, vel = pos_vel[0], pos_vel[1]
            if abs(pos) == 0:
                raise FloatingPointError(
                    "Sailing trajectory reached singularity: position relative to star is 0 m."
                )
            accel = self.solar_sail.acceleration(
                pos * m,
                self.total_mass(),
                max_accel=max_accel,
                star_mass=star_mass,
                stellar_luminosity=stellar_luminosity,
                include_gravity=include_gravity
            ) / (m / s ** 2)
            if not np.isfinite(accel):
                raise FloatingPointError(
                    f"Non-finite acceleration computed in sailing integrand: {accel}"
                )
            derivative = np.array([vel, accel], dtype=float)
            if not np.all(np.isfinite(derivative)):
                raise FloatingPointError("Non-finite derivative computed in sailing integrand.")
            return derivative

        extra_options = dict(solver_options or {})
        try:
            y_soln = scipy.integrate.solve_ivp(
                integrand,
                [0, max_sail_time_s],
                [rel_pos_m, init_v_ms],
                rtol=rtol,
                atol=atol,
                method=method,
                max_step=max_step,
                **extra_options
            )
        except FloatingPointError as err:
            raise NumericalIntegrationError(
                f"Numerical integration failed due to floating point error in integrand: {err}"
            ) from err

        if not y_soln.success or y_soln.status < 0:
            raise NumericalIntegrationError(
                f"Numerical integration for solar sailing failed: {y_soln.message} "
                f"(status={y_soln.status}, method={method})"
            )

        if not np.all(np.isfinite(y_soln.y)):
            raise NumericalIntegrationError(
                "Numerical integration for solar sailing produced non-finite values in solution."
            )

        if len(y_soln.t) < 2:
            raise NumericalIntegrationError(
                "Numerical integration for solar sailing did not produce enough time steps."
            )

        return y_soln

    def _apply_sailing_trajectory(self,
                                  y_soln,
                                  *,
                                  initial_position: unit,
                                  relative_position_to_star: unit,
                                  target_velocity: unit,
                                  max_velocity: unit,
                                  generate_electricity: bool):
        """Step through the sailing trajectory solution and update ship state."""
        sailing_time = 0.0 * s
        for i in range(1, len(y_soln.t)):
            step_dt = y_soln.t[i] - y_soln.t[i - 1]
            if step_dt <= 0:
                continue
            delta_t = step_dt * s
            self.time += delta_t
            self.position = initial_position + y_soln.y[0, i] * m - relative_position_to_star
            self.velocity = y_soln.y[1, i] * (m / s)
            acceleration = (y_soln.y[1, i] - y_soln.y[1, i - 1]) / step_dt * (m / s ** 2)
            self.log_entry(
                f"year {(self.time) / yr:0.1f} - Sailing with velocity "
                f"{self.velocity / (m / s)} m/s with acceleration "
                f"{acceleration / g}g."
            )
            sailing_time = y_soln.t[i] * s
            if target_velocity / (m / s) == 0.0 \
                    and np.sign(self.velocity / c) != np.sign(y_soln.y[1, i - 1]):
                # Deceleration scenario, zero velocity achieved
                break
            if (np.sign(relative_position_to_star / m) == np.sign(self.velocity / c))\
                    and abs(self.velocity / c) > abs(target_velocity / c):
                # Acceleration scenario, target velocity achieved
                break
            if generate_electricity:
                self.generate_electricity(self.electrical_power * delta_t)

        if target_velocity / (m / s) == 0.0:
            self.velocity = 0.0 * m / s

        self.log_entry(
            f"year {(self.time) / yr:0.1f} - Finished sailing. velocity "
            f"{self.velocity / (m / s)} m/s. Traveling at "
            f"{self.velocity / max_velocity * 100:0.1f}% of maximum sail velocity. "
            f"Sailing time was "
            f"{sailing_time / (24 * 3600 * s)} days."
        )

    def sail(self,
             target_velocity: Optional[unit] = None,
             *,
             position_of_star: unit = 0.0 * m,
             max_accel: Optional[unit] = None,
             max_sail_time: unit = 14 * 24 * 3600 * s,
             generate_electricity: bool = True,
             include_gravity: bool = False,
             star_mass: Optional[unit] = None,
             stellar_luminosity: Optional[unit] = None,
             rtol: float = 1e-6,
             atol: Union[float, Sequence[float]] = 1e-9,
             method: str = 'RK45',
             max_step: float = np.inf,
             solver_options: Optional[Dict[str, Any]] = None):
        """Accelerate or decelerate using solar sails.

        Args:
            target_velocity: Optional[unit]
                Target velocity. If None, accelerates to 90% of max velocity or
                decelerates to 0 m/s depending on orientation.
            position_of_star: unit
                Position of the star. Defaults to 0.0 * m.
            max_accel: Optional[unit]
                Maximum acceleration limit for the sail.
            max_sail_time: unit
                Maximum duration for solar sailing integration.
            generate_electricity: bool
                Whether ship systems generate electricity during flight.
            include_gravity: bool
                Whether to incorporate stellar gravitational pull.
            star_mass: Optional[unit]
                Mass of the star. Defaults to source/destination star mass.
            stellar_luminosity: Optional[unit]
                Luminosity of the star. Defaults to source/destination stellar luminosity.
            rtol: float
                Relative tolerance for numerical integration. Defaults to 1e-6.
            atol: Union[float, Sequence[float]]
                Absolute tolerance for numerical integration. Defaults to 1e-9.
            method: str
                Integration method for solve_ivp (e.g. 'RK45', 'Radau', 'BDF', 'DOP853').
                Defaults to 'RK45'.
            max_step: float
                Maximum allowed step size in seconds for solve_ivp. Defaults to np.inf.
            solver_options: Optional[Dict[str, Any]]
                Additional keyword options passed directly to scipy.integrate.solve_ivp.
        """
        if self.solar_sail is None:
            raise RuntimeError("This starship has no solar sail. Cannot sail.")

        resolved_star_mass, luminosity = self._resolve_star_properties(
            position_of_star, star_mass, stellar_luminosity)

        relative_position_to_star = self.position - position_of_star
        if relative_position_to_star / m == 0:
            raise ValueError(
                "Initial position cannot coincide with star position (singularity at star center)."
            )

        max_velocity = self.solar_sail.final_velocity(
            self.total_mass() - self.solar_sail.sail_mass,
            relative_position_to_star,
            star_mass=resolved_star_mass,
            stellar_luminosity=luminosity,
            include_gravity=include_gravity)
        if target_velocity is None:
            if np.sign(relative_position_to_star / m) == -1 * np.sign(self.velocity / c):
                # Deceleration scenario
                target_velocity = 0.0 * m / s
            else:
                target_velocity = 0.90 * max_velocity * np.sign(relative_position_to_star / m)
        if abs(target_velocity / c) > abs(max_velocity / c):
            raise ValueError(f"Unable to achieve velocity {target_velocity / c}c "
                             f"through sailing. Maximum achievable velocity"
                             f" is {max_velocity / c}c.")

        initial_position = self.position
        y_soln = self._integrate_sail(
            relative_position_to_star,
            max_sail_time=max_sail_time,
            max_accel=max_accel,
            star_mass=resolved_star_mass,
            stellar_luminosity=luminosity,
            include_gravity=include_gravity,
            rtol=rtol,
            atol=atol,
            method=method,
            max_step=max_step,
            solver_options=solver_options
        )

        self._apply_sailing_trajectory(
            y_soln,
            initial_position=initial_position,
            relative_position_to_star=relative_position_to_star,
            target_velocity=target_velocity,
            max_velocity=max_velocity,
            generate_electricity=generate_electricity
        )

    def swim(self,
             power_delivered: unit,
             swim_time: unit,
             direction: int = 1,
             generate_electricity: bool = True,
             *,
             rtol: float = 1e-6,
             atol: Union[float, Sequence[float]] = 1e-9,
             method: str = 'RK45',
             max_step: float = np.inf,
             solver_options: Optional[Dict[str, Any]] = None):
        """Accelerate or decelerate using the SWIMMER engine.

        Args:
            power_delivered: unit
                Power delivered to the SWIMMER pusher.
            swim_time: unit
                Duration of the swimming phase.
            direction: int
                Direction of thrust (+1 or -1).
            generate_electricity: bool
                Whether to generate electricity during flight.
            rtol: float
                Relative tolerance for numerical integration. Defaults to 1e-6.
            atol: Union[float, Sequence[float]]
                Absolute tolerance for numerical integration. Defaults to 1e-9.
            method: str
                Integration method for solve_ivp. Defaults to 'RK45'.
            max_step: float
                Maximum allowed step size in seconds. Defaults to np.inf.
            solver_options: Optional[Dict[str, Any]]
                Additional keyword options passed directly to scipy.integrate.solve_ivp.
        """
        if self.swimmer is None:
            raise RuntimeError("This starship has no SWIMMER engine. Cannot swim.")

        swim_time_s = swim_time / s
        if swim_time_s <= 0:
            raise ValueError(f"swim_time must be positive, got {swim_time_s} s.")

        init_pos_m = self.position / m
        init_vel_ms = self.velocity / (m / s)
        if not np.isfinite(init_pos_m) or not np.isfinite(init_vel_ms):
            raise ValueError("Initial position and velocity must be finite values.")

        def integrand(_, pos_vel):
            if not np.all(np.isfinite(pos_vel)):
                raise FloatingPointError("Non-finite state encountered in SWIMMER integrand.")
            vel = pos_vel[1]
            braking = np.sign(vel) != direction
            accel = self.swimmer.acceleration(
                power_delivered,
                abs(vel) * m / s,
                self.total_mass(),
                braking=braking
            ) / (m / s ** 2)
            if not np.isfinite(accel):
                raise FloatingPointError(
                    f"Non-finite acceleration computed in SWIMMER integrand: {accel}"
                )
            accel *= direction
            derivative = np.array([vel, accel], dtype=float)
            if not np.all(np.isfinite(derivative)):
                raise FloatingPointError("Non-finite derivative computed in SWIMMER integrand.")
            return derivative

        extra_options = dict(solver_options or {})
        try:
            y_soln = scipy.integrate.solve_ivp(
                integrand,
                [0, swim_time_s],
                [init_pos_m, init_vel_ms],
                rtol=rtol,
                atol=atol,
                method=method,
                max_step=max_step,
                **extra_options
            )
        except FloatingPointError as err:
            raise NumericalIntegrationError(
                f"Numerical integration failed due to floating point error in integrand: {err}"
            ) from err

        if not y_soln.success or y_soln.status < 0:
            raise NumericalIntegrationError(
                f"Numerical integration for SWIMMER failed: {y_soln.message} "
                f"(status={y_soln.status}, method={method})"
            )

        if not np.all(np.isfinite(y_soln.y)):
            raise NumericalIntegrationError(
                "Numerical integration for SWIMMER produced non-finite values in solution."
            )

        if len(y_soln.t) < 2:
            raise NumericalIntegrationError(
                "Numerical integration for SWIMMER did not produce enough time steps."
            )

        for i in range(1, len(y_soln.t)):
            step_dt = y_soln.t[i] - y_soln.t[i - 1]
            if step_dt <= 0:
                continue
            delta_t = step_dt * s
            self.time += delta_t
            self.position = y_soln.y[0, i] * m
            self.velocity = y_soln.y[1, i] * (m / s)
            acceleration = (y_soln.y[1, i] - y_soln.y[1, i - 1]) / step_dt * (m / s ** 2)
            self.log_entry(
                f"year {(self.time) / yr:0.1f} - Swimming with velocity "
                f"{self.velocity / (m / s)} m/s with acceleration "
                f"{acceleration / g}g."
            )
            swim_time_val = y_soln.t[i] * s
            if generate_electricity:
                self.generate_electricity(self.electrical_power * delta_t)

        self.log_entry(
            f"year {(self.time) / yr:0.1f} - Finished swimming. velocity "
            f"{self.velocity / (m / s)} m/s. "
            f"Swimming time was "
            f"{swim_time_val / (24 * 3600 * s)} days."
        )

    def print_history(self):
        """Print mission logs and messages"""
        for log, message in zip(self.history, self.log_messages):
            log = log.copy()

            print()
            print(message)
            print(log)

    def parse_logs(self):
        """Parse positions, velocities, fuels, and times from logs"""
        positions = []
        velocities = []
        fuels = []
        times = []
        for log in self.history:
            positions.append(log['position'] / ly)
            velocities.append(log['velocity'] / c)
            fuels.append(log['fuel_mass'] / kg)
            times.append(log['time'] / yr)
        return positions, velocities, fuels, times

    def plot_history(self, show_destination: bool = True):
        """Return a matplotlib figure showing mission history"""
        positions, velocities, fuels, times = self.parse_logs()
        fig = plt.figure(figsize=(12, 12))
        plt.subplot(311)
        plt.plot(times, velocities)
        plt.xlabel('Time (years)')
        plt.ylabel('Velocity (c)')
        plt.subplot(312)
        plt.plot(times, fuels)
        plt.xlabel('Time (years)')
        plt.ylabel('Fuel Mass (kg)')
        plt.subplot(313)
        plt.plot(times, positions)
        plt.xlabel('Time (years)')
        plt.ylabel('Position (light years)')
        if show_destination:
            plt.hlines(self.destination_distance / ly,
                       min(times),
                       max(times),
                       label='Destination',
                       linestyles='dashed')
        plt.legend()
        return fig
