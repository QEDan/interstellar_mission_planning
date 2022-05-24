"""Starship class"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from scimath.units.length import kilometers as km, astronomical_unit
from scimath.units.length import light_year as ly
from scimath.units.length import meters as m
from scimath.units.mass import kilograms as kg
from scimath.units.time import seconds as s
from scimath.units.time import years as yr
from scimath.units import unit

from src.solar_sail import SolarSail
from src.constants import c, g


class Starship:
    """A Starship that uses engines to accelerate."""
    def __init__(self,
                 payload_mass,
                 engines: dict,
                 initial_velocity: unit = 0 * m / s,
                 initial_position: unit = 0 * km,
                 initial_time: unit = 0 * s,
                 destination_distance: unit = 4.244 * ly,
                 solar_sail: SolarSail = None
                 ):
        self.payload_mass = payload_mass
        self.engines = engines
        self.velocity = initial_velocity
        self.position = initial_position
        self.time = initial_time
        self.destination_distance = destination_distance
        self.solar_sail = solar_sail
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
        return total_mass

    def fuel_mass(self):
        """Return the current fuel mass"""
        mass = sum([(e.fuel_mass / kg) for e in self.engines.values()]) * kg
        return mass

    def accelerate(self,
                   engine_name: str = 'main',
                   target_velocity: unit = 0 * km / s,
                   fuel_mass: unit = None,
                   decelerate: bool = False,
                   acceleration: unit = g) -> unit:
        """Accelerate the ship by burning a specified quantity of fuel.

        If no fuel mass is specified, a target_velocity should be specified instead.

        Args:
            engine_name: src
                Name of the engine
            target_velocity: unit (speed)
                Target velocity to accelerate to.
            fuel_mass: unit (mass)
                Amount of fuel mass to burn
            decelerate: bool
                If True, acceleration is toward the origin.
                If False, acceleration is toward the destination

        Returns:
            unit (speed)
                New velocity of the starship
        """
        if fuel_mass is not None:
            delta_v = self.engines[engine_name].burn_fuel(fuel_mass, self.total_mass())
            delta_t = np.abs(delta_v) / acceleration
            self.time += delta_t
            if decelerate:
                delta_pos = self.velocity * delta_t - 0.5 * acceleration * delta_t ** 2
                self.velocity -= delta_v
            else:
                delta_pos = self.velocity * delta_t + 0.5 * acceleration * delta_t ** 2
                self.velocity += delta_v
            self.time += np.abs(delta_v) / acceleration
            self.position += delta_pos

        else:
            self.engines[engine_name].set_target_delta_v(
                self.velocity - target_velocity, self.total_mass())
            delta_v = target_velocity - self.velocity
            delta_t = np.abs(delta_v) / acceleration
            if decelerate:
                delta_pos = self.velocity * delta_t - 0.5 * acceleration * delta_t ** 2
            else:
                delta_pos = self.velocity * delta_t + 0.5 * acceleration * delta_t ** 2
            self.time += delta_t
            self.velocity = target_velocity
            self.position += delta_pos

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

    def cruise(self, distance: unit):
        """Travel at current velocity without acceleration"""
        if self.velocity == 0:
            raise ValueError("The starship is not moving. Can't cruise.")
        delta_pos = distance * np.sign(self.velocity / (m / s))
        delta_t = np.abs(distance / self.velocity)
        self.position += delta_pos
        self.time += delta_t
        self.log_entry(
            f"year {(self.time - delta_t) / yr:0.1f} - "
            f"Cruise: {delta_t / yr:0.2e} years to complete. "
            f"Distance={distance / ly:0.2e} lightyears")

    def wait(self, time: unit):
        """Pass time with no acceleration"""
        self.time += time
        distance = self.velocity * time
        self.position += distance
        self.log_entry(
            f"year {(self.time - time) / yr:0.1f} - Waited: {time / yr:0.2e} years. "
            f"Distance={distance / ly:0.2e} lightyears")

    def sail(self,
             target_velocity: unit,
             relative_position_to_star: unit = astronomical_unit,
             max_accel: unit = None,
             max_sail_time: unit = 14 * 24 * 3600 * s):
        """Accelerate or decelerate using solar sails."""
        if self.solar_sail is None:
            raise RuntimeError("This starship has no solar sail. Cannot sail.")
        max_velocity = self.solar_sail.final_velocity(
            self.total_mass() - self.solar_sail.sail_mass,
            relative_position_to_star)
        if target_velocity is None:
            if relative_position_to_star / m < 0.0:
                target_velocity = 0.0 * m / s
            else:
                target_velocity = 0.90 * max_velocity
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
            if max_accel and abs(accel / (m / s ** 2)) > max_accel / (m / s ** 2):
                accel = np.sign(accel) * max_accel
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
            self.time += (y_soln.t[i] - y_soln.t[i - 1]) * s
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
            if (relative_position_to_star / m < 0 and self.velocity / c < target_velocity / c) \
                or (relative_position_to_star / m > 0 and
                    self.velocity / c > target_velocity / c):
                break
        if target_velocity / (m / s ** 2) == 0.0:
            self.velocity = 0.0 * m / s
        self.log_entry(
            f"year {(self.time) / yr:0.1f} - Finished sailing. velocity "
            f"{self.velocity / (m / s)} m/s. Traveling at "
            f"{self.velocity / max_velocity * 100:0.1f}% of maximum sail velocity. "
            f"Sailing time was "
            f"{sailing_time / (24 * 3600 * s)} days."
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

    def plot_history(self):
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
        plt.hlines(self.destination_distance / ly,
                   min(times),
                   max(times),
                   label='Destination',
                   linestyles='dashed')
        plt.legend()
        return fig
