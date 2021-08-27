"""An engine to accelerate a Starship"""

import numpy as np
from scimath.units.length import kilometers as km
from scimath.units.mass import kilograms as kg
from scimath.units.time import seconds as s


class Engine:
    """An engine to accelerate a Starship.

    This assumes constant accelerations and nonrelativistic speeds.
    """

    def __init__(self,
                 fuel_mass,
                 exhaust_velocity=500 * km / s,
                 ):
        self.fuel_mass = fuel_mass
        self.exhaust_velocity = exhaust_velocity

    def burn_fuel(self, burnt_fuel_mass, starship_mass):
        """Return the change in velocity after burning
        burnt_fuel_mass for payload starship_mass."""
        total_mass = starship_mass + self.fuel_mass
        if burnt_fuel_mass / kg > self.fuel_mass / kg:
            raise ValueError(f"Not enough fuel for this maneuver. "
                             f"Requested {burnt_fuel_mass} of {self.fuel_mass}.")
        delta_v = self.exhaust_velocity * np.log(total_mass / (total_mass - burnt_fuel_mass))
        self.fuel_mass -= burnt_fuel_mass
        return delta_v

    def set_target_delta_v(self, delta_v, starship_mass):
        """Burn fuel to achieve a specific change in velocity"""
        final_mass = starship_mass * np.exp(-1 * np.abs(delta_v) / self.exhaust_velocity)
        delta_fuel_mass = starship_mass - final_mass
        _ = self.burn_fuel(delta_fuel_mass, starship_mass)
        if self.fuel_mass / kg < 0:
            raise ValueError(
                f"Note enough fuel for this maneuver. Requested "
                f"{delta_fuel_mass} of {self.fuel_mass + delta_fuel_mass}.")
