"""Utility functions for mission planning"""
# pylint: disable=cyclic-import
from src.constants import G, solar_mass


def v_escape_solar(departure_distance):
    """Solar escape velocity"""
    escape_velocity = (2 * G * solar_mass / departure_distance)**0.5
    return escape_velocity
