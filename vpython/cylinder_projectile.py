"""Simulate projectile motion in artificial gravity.

We assume an O'Neil cylinder style habitat. The coordinate system
is fixed to a point on the inner surface of the cylinder. x, y is 
a plane tangential to the surface. x points along the long axis of the 
cylinder. z points toward the axis of rotation. We assume that
distances are much smaller than the radius of the cylinder.
The result is projectile motion as observed by an occupant of the 
habitat.
"""

from vpython import *
from copy import copy
import numpy as np
scene.forward = vector(1,0,0)
scene.up = vector(0,0,1)

# Constants
cylinder_radius = 3200  # meters
rotation_period = 114  # seconds
pi = 3.14159

# Derived quantities
angular_velocity = vector(2 * pi / rotation_period, 0, 0)

# Apparent accelerations
def centrifugal_accel(radius):
    """Cetrifugal acceleration for scalar radius"""
    accel = radius * (2 * pi / rotation_period)**2
    return accel * vector(0, 0, -1)

def coriolis_accel(velocity):
    """Coriolis acceleration for the given velocity.

    Args:
        velocity: vector, velocity relative to rotation frame, meters

    Returns:
        acceleration: vector, meters / second**2
    """
    accel = -2 * angular_velocity.cross(vector(0,0,velocity.z))
    return accel

def displacement(pos, p_init):
    x1 = 0
    y1 = 0
    x2 = p_init.x
    y2 = p_init.y
    x0 = pos.x
    y0 = pos.y
    distance = np.abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / \
               np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

ground = box(pos=vector(0,0,0), size=vector(90,90,0.5), color=color.green)

z_graph = graph(xtitle='time (s)', ytitle='height (m)')
z_plot = gcurve(color=color.red)
z_plot_earth = gcurve(color=color.yellow)

d_graph = graph(xtitle='time (s)', ytitle='coriolis displacement (m)')
d_plot = gcurve(color=color.red)

a_cent_graph = graph(xtitle='time (s)', ytitle='centrifugal acceleration (m/s^2)')
a_cent_plot = gcurve(color=color.red)

a_cor_graph = graph(xtitle='time (s)', ytitle='coriolis acceleration (m/s^2)')
a_cor_plot = gcurve(color=color.red)

p_init = vector(1, 1, 50) * 0.145
ball = sphere(pos=vector(0,0,0.01), radius=0.230, color=color.red, 
                make_trail=True, trail_radius=0.5,
              trail_type='points', interval=10, retain=500)
ball.mass = 0.145
ball.p = p_init

earth_ball = sphere(pos=copy(ball.pos), radius=copy(ball.radius),
                    color=color.yellow, 
                    make_trail=True, trail_type='points',
                    trail_radius=0.5,
                    interval=10, retain=500)
earth_ball.mass = copy(ball.mass)
ball.p = p_init
earth_ball.p = p_init

dt = 1e-3
not_landed = True
t = 0.0

while not_landed:
    rate(200)
    r = cylinder_radius - ball.pos.z
    a_cent = centrifugal_accel(r)
    a_cor = coriolis_accel(ball.p / ball.mass)
    ball.p = ball.p + ball.mass * (
        a_cent + a_cor) * dt 
    ball.pos = ball.pos + (ball.p / ball.mass) * dt
    earth_ball.p = earth_ball.p + earth_ball.mass * vector(0,0,-9.81) * dt
    earth_ball.pos = earth_ball.pos + (earth_ball.p / earth_ball.mass) * dt
    not_landed = r <  cylinder_radius
    t += dt
    z_plot.plot(t, ball.pos.z)
    z_plot_earth.plot(t, earth_ball.pos.z)
    d_plot.plot(t, displacement(ball.pos, p_init)) 
    a_cent_plot.plot(t, a_cent.mag)
    a_cor_plot.plot(t, a_cor.mag)
print('Done!')
