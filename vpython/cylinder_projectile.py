"""Simulate projectile motion in artificial gravity.

We assume an O'Neil cylinder style habitat. The coordinate system
is fixed to a point on the inner surface of the cylinder. x, y is 
a plane tangential to the surface. x points along the long axis of the 
cylinder. z points toward the axis of rotation.
The result is projectile motion as observed by an occupant of the 
habitat.
"""

from vpython import *
from copy import copy
scene.caption = """In GlowScript programs:
To rotate "camera", drag with right button or Ctrl-drag.
To zoom, drag with middle button or Alt/Option depressed, or use scroll wheel.
  On a two-button mouse, middle is left + right.
To pan left/right and up/down, Shift-drag.
Touch screen: pinch/extend to zoom, swipe or two-finger rotate."""
scene.forward = vector(1,0,0)
scene.up = vector(0,0,1)

# Constants
cylinder_radius = 3200  # meters
rotation_period = 114  # seconds
pi = 3.14159

# Derived quantities
angular_velocity = vector(2 * pi / rotation_period, 0, 0)

# Apparent accelerations
def centripedal_accel(radius):
    """Cetripedal acceleration for scalar radius"""
    accel = radius * (2 * pi / rotation_period)**2
    return accel * vector(0, 0, -1)

def coriolis_accel(velocity):
    """Coriolis acceleration for the given velocity.

    Args:
        velocity: vector, velocity relative to rotation frame, meters

    Returns:
        acceleration: vector, meters / second**2
    """
    accel = -2 * angular_velocity.cross(velocity)
    return accel

ground = box(pos=vector(0,0,0), size=vector(30,30,0.5), color=color.green)

# xaxis = PhysAxis(ground, 10)
# yaxis = PhysAxis(ground, 5)




ball = sphere(pos=vector(0,0,1.0), radius=0.230, color=color.red, 
                make_trail=True, trail_type='points', interval=10, retain=500)
ball.mass = 0.145
ball.p = vector(1, 1, 1)

earth_ball = sphere(pos=copy(ball.pos), radius=copy(ball.radius),
                    color=color.yellow, 
                    make_trail=True, trail_type='points',
                    interval=10, retain=500)
earth_ball.mass = copy(ball.mass)
ball.p = vector(1, 1, 3)
earth_ball.p = copy(ball.p)

dt = 1e-3
not_landed = True
while not_landed:
    rate(200)
    r = cylinder_radius - ball.pos.z
    ball.p = ball.p + ball.mass * (
        centripedal_accel(r) + coriolis_accel(ball.p / ball.mass)) * dt 
    ball.pos = ball.pos + (ball.p / ball.mass) * dt
    earth_ball.p = earth_ball.p + earth_ball.mass * vector(0,0,-9.81) * dt
    earth_ball.pos = earth_ball.pos + (earth_ball.p / earth_ball.mass) * dt
    not_landed = r <  cylinder_radius

print('Done!')
