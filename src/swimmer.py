"""A SWIMMER engine to accelerate and decelerate a starship using plasma momentum exchange.

This engine is propellantless, requiring only electrical power which can be beamed or
produced on board.

Reference:
    Spacecraft With Interstellar Medium Momentum Exchange Reactions: The potential and
    limitations of propellantless interstellar travel
    By Drew Brisbin
    https://arxiv.org/abs/1808.02019
"""

from typing import Optional
import numpy as np
from scimath.units.unit import unit
from scimath.units.length import meters as m, astronomical_unit as au, light_year as ly
from scimath.units.time import seconds as s
from scimath.units.mass import kilograms as kg
from scimath.units.temperature import kelvin as K
from scimath.units.SI import watt

# Physical constants in SI
EPSILON_0 = 8.8541878128e-12  # F / m = C^2 / (J * m)
K_BOLTZMANN = 1.380649e-23     # J / K
E_CHARGE = 1.602176634e-19     # C
PROTON_MASS_KG = 1.67262192369e-27  # kg


class ISMEnvironment:
    """Interstellar Medium (ISM) or Astrospheric plasma and gas environment."""

    def __init__(self,
                 name: str,
                 ion_density: unit,
                 neutral_density: unit = 0.0 / (m ** 3),
                 temperature: unit = 7000.0 * K,
                 streaming_velocity: unit = 0.0 * m / s,
                 ion_mass: unit = PROTON_MASS_KG * kg):
        self.name = name
        self.ion_density = ion_density
        self.neutral_density = neutral_density
        self.temperature = temperature
        self.streaming_velocity = streaming_velocity
        self.ion_mass = ion_mass

    @property
    def total_number_density(self) -> unit:
        """Total particle number density (ions + neutrals)."""
        return self.ion_density + self.neutral_density

    @property
    def ionization_fraction(self) -> float:
        """Fraction of particles that are ionized."""
        n_tot = (self.ion_density + self.neutral_density) / (1.0 / (m ** 3))
        if n_tot <= 0:
            return 1.0
        return float((self.ion_density / (1.0 / (m ** 3))) / n_tot)

    def ion_mass_density(self) -> unit:
        """Mass density of ions in the medium."""
        return self.ion_density * self.ion_mass

    def neutral_mass_density(self) -> unit:
        """Mass density of neutral gas in the medium."""
        return self.neutral_density * self.ion_mass

    def total_mass_density(self) -> unit:
        """Total mass density of ions and neutral gas."""
        return (self.ion_density + self.neutral_density) * self.ion_mass

    def debye_length(self) -> unit:
        """Electron Debye length in the plasma."""
        n_i_m3 = max(1e-12, float(self.ion_density / (1.0 / (m ** 3))))
        t_kelvin = max(1.0, float(self.temperature / K))
        lambda_d_val = np.sqrt(EPSILON_0 * K_BOLTZMANN * t_kelvin / (n_i_m3 * (E_CHARGE ** 2)))
        return lambda_d_val * m

    def thermal_velocity(self) -> unit:
        """Mean thermal speed of protons in the plasma."""
        t_kelvin = max(1.0, float(self.temperature / K))
        m_kg = float(self.ion_mass / kg)
        v_th_val = np.sqrt(8.0 * K_BOLTZMANN * t_kelvin / (np.pi * m_kg))
        return v_th_val * (m / s)

    @classmethod
    def hot_ionized(cls) -> 'ISMEnvironment':
        """Hot Ionized Medium (HIM) / Local Bubble coronal gas: n ~ 0.005 cm^-3, T ~ 1e6 K."""
        return cls(
            name="Hot Ionized Medium (Local Bubble)",
            ion_density=0.005 / ((m / 100) ** 3),
            neutral_density=0.0 / (m ** 3),
            temperature=1.0e6 * K,
            streaming_velocity=0.0 * m / s
        )

    @classmethod
    def local_cloud(cls) -> 'ISMEnvironment':
        """Local Interstellar Cloud (LIC): n_i ~ 0.07 cm^-3, n_n ~ 0.10 cm^-3, T ~ 7000 K."""
        return cls(
            name="Local Interstellar Cloud (LIC)",
            ion_density=0.07 / ((m / 100) ** 3),
            neutral_density=0.10 / ((m / 100) ** 3),
            temperature=7000.0 * K,
            streaming_velocity=0.0 * m / s
        )

    @classmethod
    def warm_ionized(cls) -> 'ISMEnvironment':
        """Warm Ionized Medium (WIM): n_i ~ 0.3 cm^-3, n_n ~ 0.03 cm^-3, T ~ 8000 K."""
        return cls(
            name="Warm Ionized Medium (WIM)",
            ion_density=0.30 / ((m / 100) ** 3),
            neutral_density=0.03 / ((m / 100) ** 3),
            temperature=8000.0 * K,
            streaming_velocity=0.0 * m / s
        )

    @classmethod
    def warm_neutral(cls) -> 'ISMEnvironment':
        """Warm Neutral Medium (WNM): n_i ~ 0.03 cm^-3, n_n ~ 0.5 cm^-3, T ~ 6000 K."""
        return cls(
            name="Warm Neutral Medium (WNM)",
            ion_density=0.03 / ((m / 100) ** 3),
            neutral_density=0.50 / ((m / 100) ** 3),
            temperature=6000.0 * K,
            streaming_velocity=0.0 * m / s
        )

    @classmethod
    def cold_neutral(cls) -> 'ISMEnvironment':
        """Cold Neutral Medium (CNM): n_i ~ 0.04 cm^-3, n_n ~ 30 cm^-3, T ~ 100 K."""
        return cls(
            name="Cold Neutral Medium (CNM)",
            ion_density=0.04 / ((m / 100) ** 3),
            neutral_density=30.0 / ((m / 100) ** 3),
            temperature=100.0 * K,
            streaming_velocity=0.0 * m / s
        )

    @classmethod
    def molecular_cloud(cls, density_cm3: float = 1000.0) -> 'ISMEnvironment':
        """Dense / Diffuse Molecular Cloud: n ~ 10^2 - 10^4 cm^-3, T ~ 20 K."""
        ion_density_cm3 = max(0.001, density_cm3 * 1.0e-5)
        return cls(
            name=f"Molecular Cloud ({density_cm3:0.0f} cm^-3)",
            ion_density=ion_density_cm3 / ((m / 100) ** 3),
            neutral_density=density_cm3 / ((m / 100) ** 3),
            temperature=20.0 * K,
            streaming_velocity=0.0 * m / s
        )

    @classmethod
    def heliosphere(cls, distance_from_star: unit = 1.0 * au) -> 'ISMEnvironment':
        """Stellar wind plasma in a star's heliosphere / astrosphere."""
        dist_au = max(0.05, float(distance_from_star / au))
        n_i_cm3 = 7.0 / (dist_au ** 2) + 0.07
        n_n_cm3 = 0.05
        t_k = 1.4e5 / np.sqrt(dist_au) + 7000.0
        return cls(
            name=f"Heliosphere ({dist_au:.1f} AU)",
            ion_density=n_i_cm3 / ((m / 100) ** 3),
            neutral_density=n_n_cm3 / ((m / 100) ** 3),
            temperature=t_k * K,
            streaming_velocity=450.0 * 1000.0 * (m / s)
        )


class TripProfile:
    """Plasma and gas environment profile for a trip from Sol to a destination star."""

    def __init__(self,
                 destination_distance: unit = 4.244 * ly,
                 sol_heliopause: unit = 120.0 * au,
                 dest_astropause: unit = 120.0 * au,
                 sol_wind_speed: unit = 450.0 * 1000.0 * (m / s),
                 dest_wind_speed: unit = 450.0 * 1000.0 * (m / s),
                 interstellar_env: Optional[ISMEnvironment] = None):
        self.destination_distance = destination_distance
        self.sol_heliopause = sol_heliopause
        self.dest_astropause = dest_astropause
        self.sol_wind_speed = sol_wind_speed
        self.dest_wind_speed = dest_wind_speed
        self.interstellar_env = interstellar_env or ISMEnvironment.local_cloud()

    def environment_at(self, position: unit) -> ISMEnvironment:
        """Return the plasma environment at a given position along the Sol -> destination axis."""
        pos_m = float(position / m)
        dest_m = float(self.destination_distance / m)
        sol_hp_m = float(self.sol_heliopause / m)
        dest_ap_m = float(self.dest_astropause / m)
        au_m = float(au / m)

        # Region 1: Inside Sol's heliosphere
        if pos_m < sol_hp_m:
            r_au = max(0.1, pos_m / au_m)
            n_i = (7.0 / (r_au ** 2) + 0.07) / ((m / 100) ** 3)
            t_k = (1.4e5 / np.sqrt(r_au) + 7000.0) * K
            return ISMEnvironment(
                name=f"Sol Heliosphere ({r_au:.1f} AU)",
                ion_density=n_i,
                neutral_density=0.05 / ((m / 100) ** 3),
                temperature=t_k,
                streaming_velocity=self.sol_wind_speed
            )

        # Region 3: Inside destination astrosphere
        dist_to_dest_m = dest_m - pos_m
        if dist_to_dest_m < dest_ap_m:
            r_au = max(0.1, dist_to_dest_m / au_m)
            n_i = (7.0 / (r_au ** 2) + 0.07) / ((m / 100) ** 3)
            t_k = (1.4e5 / np.sqrt(r_au) + 7000.0) * K
            return ISMEnvironment(
                name=f"Destination Astrosphere ({r_au:.1f} AU)",
                ion_density=n_i,
                neutral_density=0.05 / ((m / 100) ** 3),
                temperature=t_k,
                streaming_velocity=-self.dest_wind_speed
            )

        # Region 2: Interstellar cruise
        return self.interstellar_env

    def density_at(self, position: unit) -> unit:
        """Return the ion number density at a given position."""
        return self.environment_at(position).ion_density

    def is_interstellar(self, position: unit) -> bool:
        """Check if position is in the interstellar cruise phase outside astrospheres."""
        pos_m = float(position / m)
        dest_m = float(self.destination_distance / m)
        sol_hp_m = float(self.sol_heliopause / m)
        dest_ap_m = float(self.dest_astropause / m)
        return sol_hp_m <= pos_m <= (dest_m - dest_ap_m)


class Swimmer:
    """A SWIMMER engine to accelerate and decelerate a starship."""

    def __init__(self,
                 pusher_area: unit,
                 pusher_areal_density: unit = 1.0 * kg / (m ** 2),
                 ion_mass: unit = PROTON_MASS_KG * kg,
                 ion_density: unit = 0.07 / ((m / 100) ** 3),
                 neutral_density: unit = 0.10 / ((m / 100) ** 3),
                 plasma_temperature: unit = 7000.0 * K,
                 physical_area: Optional[unit] = None,
                 hull_area: unit = 100.0 * (m ** 2),
                 drag_coefficient: float = 2.0,
                 environment: Optional[ISMEnvironment] = None,
                 trip_profile: Optional[TripProfile] = None,
                 scale_area_with_debye: bool = True):
        self.pusher_area = pusher_area
        self.pusher_areal_density = pusher_areal_density
        self.ion_mass = ion_mass
        self.ion_density = ion_density
        self.neutral_density = neutral_density
        self.plasma_temperature = plasma_temperature
        self.hull_area = hull_area
        self.drag_coefficient = drag_coefficient
        self.environment = environment
        self.trip_profile = trip_profile
        self.scale_area_with_debye = scale_area_with_debye

        # Reference Debye length at standard LIC conditions (T = 7000 K, n = 0.07 cm^-3)
        self.nominal_debye_length = 21.82255 * m

        # Physical wire area: for 30 um wire diameter (Brisbin 2018 Sec. 4)
        # Ratio of wire physical cross section to 2 * lambda_D interaction sheath:
        # 30e-6 m / (2 * 21.82 m) ~ 6.87e-7
        if physical_area is not None:
            self.physical_area = physical_area
        else:
            wire_fraction = 30.0e-6 / (2.0 * float(self.nominal_debye_length / m))
            self.physical_area = self.pusher_area * wire_fraction + self.hull_area

    def shed_area(self, delta_area: unit):
        """Remove specified amount of area from the pusher plate to shed mass."""
        self.pusher_area -= delta_area
        wire_fraction = 30.0e-6 / (2.0 * float(self.nominal_debye_length / m))
        self.physical_area = self.pusher_area * wire_fraction + self.hull_area

    def pusher_mass(self) -> unit:
        """Return the mass of the pusher plate."""
        return self.pusher_area * self.pusher_areal_density

    def get_environment(self,
                        environment: Optional[ISMEnvironment] = None,
                        position: Optional[unit] = None,
                        ion_density: Optional[unit] = None) -> ISMEnvironment:
        """Resolve the active ISM / plasma environment."""
        if environment is not None:
            return environment
        if position is not None and self.trip_profile is not None:
            return self.trip_profile.environment_at(position)
        if self.environment is not None:
            return self.environment

        target_density = ion_density if ion_density is not None else self.ion_density
        return ISMEnvironment(
            name="Custom Environment",
            ion_density=target_density,
            neutral_density=self.neutral_density,
            temperature=self.plasma_temperature,
            streaming_velocity=0.0 * (m / s),
            ion_mass=self.ion_mass
        )

    def debye_length(self, environment: Optional[ISMEnvironment] = None) -> unit:
        """Compute the electron Debye length in the current environment."""
        env = self.get_environment(environment=environment)
        return env.debye_length()

    def effective_area(self, environment: Optional[ISMEnvironment] = None) -> unit:
        """Calculate effective interaction cross section of the pusher plate in plasma.

        Tether-based pusher plates interact with plasma ions passing within ~lambda_D
        (Brisbin 2018 Eq. 6). When Debye length changes across environments, effective area scales.
        """
        if not self.scale_area_with_debye:
            return self.pusher_area

        local_lambda_d = self.debye_length(environment=environment)
        ratio = float(local_lambda_d / self.nominal_debye_length)
        # Bound scaling between 0.01x (dense cold clouds) and 50x (hot ionized bubbles)
        ratio = max(0.01, min(50.0, ratio))
        return self.pusher_area * ratio

    def stopping_potential(self, velocity: unit) -> unit:
        """Stopping potential required to reflect ions: phi_stop = m_ion * v^2 / (2 * e)."""
        v_ms = float(abs(velocity) / (m / s))
        m_kg = float(self.ion_mass / kg)
        energy_joules = 0.5 * m_kg * (v_ms ** 2)
        voltage = energy_joules / E_CHARGE
        # 1 V = 1 kg * m^2 / (s^3 * A) = 1 J / C
        volts_unit = (kg * (m ** 2)) / ((s ** 3) * (watt / (kg * (m ** 2) / (s ** 3))))
        return voltage * volts_unit

    def drag_force(self,
                   velocity: unit,
                   environment: Optional[ISMEnvironment] = None,
                   position: Optional[unit] = None) -> unit:
        """Compute parasitic drag force from neutral gas and hull ion collisions."""
        env = self.get_environment(environment=environment, position=position)
        v_rel = abs(velocity - env.streaming_velocity)
        v_rel_ms = float(v_rel / (m / s))
        if v_rel_ms <= 0:
            return 0.0 * kg * m / (s ** 2)

        rho_neutral = float(env.neutral_mass_density() / (kg / (m ** 3)))
        rho_ion = float(env.ion_mass_density() / (kg / (m ** 3)))

        a_phys = float(self.physical_area / (m ** 2))
        a_hull = float(self.hull_area / (m ** 2))

        # Free molecular aerodynamic ram drag: 0.5 * Cd * rho * A * v^2
        f_neutral = 0.5 * self.drag_coefficient * rho_neutral * a_phys * (v_rel_ms ** 2)
        f_hull_ion = 0.5 * self.drag_coefficient * rho_ion * a_hull * (v_rel_ms ** 2)
        return (f_neutral + f_hull_ion) * (kg * m / (s ** 2))

    def net_force(self,
                  power_delivered: unit,
                  abs_velocity: unit,
                  braking: bool = False,
                  environment: Optional[ISMEnvironment] = None,
                  position: Optional[unit] = None,
                  ion_density: Optional[unit] = None) -> unit:
        """Compute net force produced by the SWIMMER engine.

        Accounts for plasma momentum exchange and parasitic drag.
        """
        env = self.get_environment(
            environment=environment, position=position, ion_density=ion_density
        )
        v_rel = abs(abs_velocity - env.streaming_velocity)
        v_rel_ms = float(v_rel / (m / s))
        v_th_ms = float(env.thermal_velocity() / (m / s))
        v_eff_ms = np.sqrt(v_rel_ms ** 2 + v_th_ms ** 2)

        a_eff = self.effective_area(environment=env)
        a_eff_m2 = float(a_eff / (m ** 2))
        rho_ion_kg_m3 = float(env.ion_mass_density() / (kg / (m ** 3)))

        dot_m = a_eff_m2 * rho_ion_kg_m3 * v_eff_ms
        p_watts = float(power_delivered / watt)

        force_unit = kg * m / (s ** 2)
        f_drag = self.drag_force(abs_velocity, environment=env)
        f_drag_n = float(f_drag / force_unit)

        if not braking:
            # Acceleration mode: Brisbin 2018 Eq. 2
            if p_watts > 0:
                f_gross = np.sqrt(dot_m * (2.0 * p_watts + dot_m * (v_rel_ms ** 2)))
                f_swimmer = f_gross - dot_m * v_rel_ms
                f_net = max(0.0, f_swimmer - f_drag_n)
            else:
                # Unpowered coasting: passive drag opposes motion
                f_net = -(f_drag_n + dot_m * v_rel_ms)
            return f_net * force_unit

        # Braking mode: Brisbin 2018 Sec. 3 & 4
        if p_watts > 0:
            f_gross = np.sqrt(dot_m * (2.0 * p_watts + dot_m * (v_rel_ms ** 2)))
            f_swimmer_brake = f_gross + dot_m * v_rel_ms
        else:
            # Passive braking (P = 0): elastic reflection limit 2 * dot_m * v_rel
            f_swimmer_brake = 2.0 * dot_m * v_rel_ms

        f_brake_total = f_swimmer_brake + f_drag_n
        return f_brake_total * force_unit

    def acceleration(self,
                     power_delivered: unit,
                     abs_velocity: unit,
                     total_mass: unit,
                     braking: bool = False,
                     environment: Optional[ISMEnvironment] = None,
                     position: Optional[unit] = None,
                     ion_density: Optional[unit] = None) -> unit:
        """Compute the acceleration from the SWIMMER engine."""
        force = self.net_force(
            power_delivered=power_delivered,
            abs_velocity=abs_velocity,
            braking=braking,
            environment=environment,
            position=position,
            ion_density=ion_density
        )
        return abs(force / total_mass)
