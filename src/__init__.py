"""Simplify imports"""
from .engine import Engine
from .solar_sail import SolarSail
from .swimmer import Swimmer, ISMEnvironment, TripProfile
from .starship import Starship, NumericalIntegrationError

__all__ = [
    'Engine',
    'SolarSail',
    'Swimmer',
    'ISMEnvironment',
    'TripProfile',
    'Starship',
    'NumericalIntegrationError'
]
