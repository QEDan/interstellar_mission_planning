"""An engine to accelerate a Starship using relativistic rocket mechanics."""

import numpy as np
from scimath.units.length import kilometers as km
from scimath.units.length import meters as m
from scimath.units.mass import kilograms as kg
from scimath.units.time import seconds as s
from src.constants import c


class Engine:
    """An engine to accelerate a Starship using relativistic rocket mechanics."""

    def __init__(self,
                 fuel_mass,
                 exhaust_velocity=500 * km / s,
                 ):
        self.fuel_mass = fuel_mass
        self.exhaust_velocity = exhaust_velocity

    def burn_fuel(self, burnt_fuel_mass, starship_mass):
        """Return the change in velocity after burning
        burnt_fuel_mass for payload starship_mass using relativistic rocket equation."""
        total_mass = starship_mass + self.fuel_mass
        if burnt_fuel_mass / kg > self.fuel_mass / kg:
            raise ValueError(f"Not enough fuel for this maneuver. "
                             f"Requested {burnt_fuel_mass} of {self.fuel_mass}.")
        m0_kg = total_mass / kg
        mf_kg = (total_mass - burnt_fuel_mass) / kg
        ve_ms = self.exhaust_velocity / (m / s)
        c_ms = c / (m / s)

        u = (ve_ms / c_ms) * np.log(m0_kg / mf_kg)
        delta_v_ms = c_ms * np.tanh(u)
        delta_v = delta_v_ms * (m / s)

        self.fuel_mass -= burnt_fuel_mass
        return delta_v

    def set_target_delta_v(self, delta_v, starship_mass):
        """Burn fuel for target velocity change using relativistic rocket equation."""
        total_mass = starship_mass + self.fuel_mass
        delta_v_ms = abs(delta_v / (m / s))
        ve_ms = self.exhaust_velocity / (m / s)
        c_ms = c / (m / s)

        if delta_v_ms >= c_ms:
            raise ValueError(f"Target delta_v {delta_v} cannot reach or exceed speed of light.")

        m0_kg = total_mass / kg
        exponent = c_ms / (2.0 * ve_ms)
        mass_ratio_term = (1.0 - delta_v_ms / c_ms) / (1.0 + delta_v_ms / c_ms)
        mf_kg = m0_kg * (mass_ratio_term ** exponent)
        delta_fuel_mass_kg = m0_kg - mf_kg
        delta_fuel_mass = delta_fuel_mass_kg * kg

        _ = self.burn_fuel(delta_fuel_mass, starship_mass)
        if self.fuel_mass / kg < 0:
            raise ValueError(
                f"Not enough fuel for this maneuver. Requested "
                f"{delta_fuel_mass} of {self.fuel_mass + delta_fuel_mass}.")
