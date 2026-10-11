from scimath.units.unit import unit
from src.constants import G, solar_mass


def v_escape(distance: unit, star_mass: unit = solar_mass) -> unit:
    """Stellar escape velocity at a given distance from a star."""
    escape_velocity = (2 * G * star_mass / distance) ** 0.5
    return escape_velocity


def v_escape_solar(departure_distance: unit) -> unit:
    """Solar escape velocity."""
    return v_escape(departure_distance, solar_mass)


def gravitational_acceleration(distance: unit, star_mass: unit = solar_mass) -> unit:
    """Gravitational acceleration from a star at a given distance."""
    return G * star_mass / (distance ** 2)


def effective_delta_v_with_gravity(target_velocity: unit,
                                   distance: unit,
                                   star_mass: unit = solar_mass) -> unit:
    """Effective delta-v required to achieve target velocity accounting for gravity."""
    v_esc = v_escape(distance, star_mass)
    return (target_velocity ** 2 + v_esc ** 2) ** 0.5
