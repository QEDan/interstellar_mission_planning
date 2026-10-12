"""Starship class"""

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


class Starship:
    """A Starship that uses engines to accelerate."""
    def __init__(self,
                 payload_mass,
                 engines: dict,
                 initial_velocity: unit = 0 * m / s,
                 initial_position: unit = 0 * km,
                 initial_time: unit = 0 * s,
                 destination_distance: unit = 4.244 * ly,
                 electrical_power: unit = 1.5e11 * watt,
                 solar_sail: SolarSail = None,
                 swimmer: Swimmer = None,
                 source_star_mass: unit = solar_mass,
                 destination_star_mass: unit = solar_mass,
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
        mass = sum([(e.fuel_mass / kg) for e in self.engines.values()]) * kg
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

    def accelerate(self,
                   engine_name: str = 'main',
                   target_velocity: unit = 0 * km / s,
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
                Whether to include gravitational escape velocity corrections
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
            if v_curr_ms != 0 and np.sign(v_curr_ms) != np.sign(direction):
                is_decelerating = True
            else:
                is_decelerating = False
        else:
            is_decelerating = abs(target_v_ms) < abs(v_curr_ms)

        if not is_decelerating:
            v_esc = self.source_escape_velocity(departure_distance)
            v_esc_ms = v_esc / (m / s)
            if fuel_mass is not None:
                delta_v_burn = self.engines[engine_name].burn_fuel(fuel_mass, self.total_mass())
                delta_v_burn_ms = delta_v_burn / (m / s)
                v_gross_ms = abs(v_curr_ms) + delta_v_burn_ms
                if include_gravity and v_gross_ms >= v_esc_ms:
                    v_net_ms = np.sqrt(v_gross_ms ** 2 - v_esc_ms ** 2)
                    self.velocity = direction * v_net_ms * (m / s)
                else:
                    self.velocity += direction * delta_v_burn
                delta_v = delta_v_burn
            else:
                v_target_mag = abs(target_v_ms)
                v_curr_mag = abs(v_curr_ms)
                if include_gravity:
                    v_req_mag = np.sqrt(v_target_mag ** 2 + v_esc_ms ** 2)
                    delta_v_burn_ms = max(0.0, v_req_mag - v_curr_mag)
                else:
                    delta_v_burn_ms = abs(v_target_mag - v_curr_mag)
                delta_v_burn = delta_v_burn_ms * (m / s)
                self.engines[engine_name].set_target_delta_v(delta_v_burn, self.total_mass())
                delta_v = delta_v_burn
                self.velocity = target_velocity

        else:
            v_esc = self.destination_escape_velocity(destination_distance_from_star)
            v_esc_ms = v_esc / (m / s)
            if fuel_mass is not None:
                delta_v_burn = self.engines[engine_name].burn_fuel(fuel_mass, self.total_mass())
                delta_v_burn_ms = delta_v_burn / (m / s)
                if include_gravity:
                    v_inbound_ms = np.sqrt(v_curr_ms ** 2 + v_esc_ms ** 2)
                    v_rem_ms = max(0.0, v_inbound_ms - delta_v_burn_ms)
                    self.velocity = direction * v_rem_ms * (m / s)
                else:
                    v_rem_ms = max(0.0, abs(v_curr_ms) - delta_v_burn_ms)
                    self.velocity = direction * v_rem_ms * (m / s)
                delta_v = delta_v_burn
            else:
                v_target_mag = abs(target_v_ms)
                v_curr_mag = abs(v_curr_ms)
                if include_gravity:
                    v_inbound_ms = np.sqrt(v_curr_mag ** 2 + v_esc_ms ** 2)
                    delta_v_burn_ms = max(0.0, v_inbound_ms - v_target_mag)
                else:
                    delta_v_burn_ms = abs(v_curr_mag - v_target_mag)
                delta_v_burn = delta_v_burn_ms * (m / s)
                self.engines[engine_name].set_target_delta_v(delta_v_burn, self.total_mass())
                delta_v = delta_v_burn
                self.velocity = target_velocity

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

    def sail(self,
             target_velocity: unit,
             position_of_star: unit = 0.0 * m,
             max_accel: unit = None,
             max_sail_time: unit = 14 * 24 * 3600 * s,
             generate_electricity: bool = True):
        """Accelerate or decelerate using solar sails."""
        if self.solar_sail is None:
            raise RuntimeError("This starship has no solar sail. Cannot sail.")
        relative_position_to_star = self.position - position_of_star
        max_velocity = self.solar_sail.final_velocity(
            self.total_mass() - self.solar_sail.sail_mass,
            relative_position_to_star)
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

        def integrand(_, pos_vel):
            pos, vel = pos_vel[0], pos_vel[1]
            if np.isnan(pos) or np.isnan(vel):
                raise ValueError("Nan values in integrand.")
            accel = self.solar_sail.acceleration(
                pos * m,
                self.total_mass(),
                max_accel=max_accel
            ) / (m / s ** 2)
            derivative = np.array([
                vel,
                accel
            ])
            return derivative

        initial_velocity = self.velocity
        y_soln = scipy.integrate.solve_ivp(
            integrand,
            [0, max_sail_time / s],
            [relative_position_to_star / m, initial_velocity / (m / s)]
        )
        initial_position = self.position
        for i in range(1, len(y_soln.t)):
            delta_t = (y_soln.t[i] - y_soln.t[i - 1]) * s
            self.time += delta_t
            self.position = initial_position + y_soln.y[0, i] * m - relative_position_to_star
            self.velocity = y_soln.y[1, i] * (m / s)
            acceleration = (y_soln.y[1, i] - y_soln.y[1, i - 1]) / (
                y_soln.t[i] - y_soln.t[i - 1]) * (m / s ** 2)
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

    def swim(self,
             power_delivered: unit,
             swim_time: unit,
             direction: int = 1,
             generate_electricity: bool = True):
        """Accelerate or decelerate using solar sails."""
        if self.swimmer is None:
            raise RuntimeError("This starship has no SWIMMER engine. Cannot swim.")

        def integrand(_, pos_vel):
            pos, vel = pos_vel[0], pos_vel[1]
            if np.isnan(pos) or np.isnan(vel):
                raise ValueError("Nan values in SWIMMER integrand.")
            braking = np.sign(vel) != direction
            accel = self.swimmer.acceleration(
                power_delivered,
                abs(vel) * m / s,
                self.total_mass(),
                braking=braking
            ) / (m / s ** 2)
            accel *= direction
            derivative = np.array([
                vel,
                accel
            ])
            return derivative

        y_soln = scipy.integrate.solve_ivp(
            integrand,
            [0, swim_time / s],
            [self.position / m, self.velocity / (m / s)]
        )
        for i in range(1, len(y_soln.t)):
            delta_t = (y_soln.t[i] - y_soln.t[i - 1]) * s
            self.time += delta_t
            self.position = y_soln.y[0, i] * m
            self.velocity = y_soln.y[1, i] * (m / s)
            acceleration = (y_soln.y[1, i] - y_soln.y[1, i - 1]) / (
                y_soln.t[i] - y_soln.t[i - 1]) * (m / s ** 2)
            self.log_entry(
                f"year {(self.time) / yr:0.1f} - Swimming with velocity "
                f"{self.velocity / (m / s)} m/s with acceleration "
                f"{acceleration / g}g."
            )
            swim_time = y_soln.t[i] * s
            if generate_electricity:
                self.generate_electricity(self.electrical_power * delta_t)

        self.log_entry(
            f"year {(self.time) / yr:0.1f} - Finished swimming. velocity "
            f"{self.velocity / (m / s)} m/s. "
            f"Swimming time was "
            f"{swim_time / (24 * 3600 * s)} days."
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
