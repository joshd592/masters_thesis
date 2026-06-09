# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.colors as mcolors
from numpy.linalg import norm

mu = 398600.4418 #km^3/s^2

def rainbow_plot2(ax, x, y, linewidth=2, **kwargs):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) < 2:
        raise ValueError("At least 2 points are required to draw a line.")

    # Rainbow spans hue 0.0 (red) → ~0.83 (purple/violet) in HSV space
    hues = np.linspace(0.0, 0.83, len(x) - 1)

    for i, hue in enumerate(hues):
        color = mcolors.hsv_to_rgb([hue, 1.0, 1.0])
        ax.plot(
            [x[i], x[i + 1]],
            [y[i], y[i + 1]],
            color=color,
            linewidth=linewidth,
            **kwargs,
        )
def rainbow_plot(ax, x=[], y=[]):
    if len(x)!=0 and len(y)!=0:
      rainbow_plot2(ax, x, y)
    else:
      rainbow_plot2(ax, range(len(x)), x)

def ECIprop(pos, vel, step, accel=None):
    if accel is None:
        accel = np.zeros((3,1))
    accel = np.asarray(accel).reshape(3,)

    halfvel = vel + (0.5*(-mu*pos*step/norm(pos)**3)) + .5*accel*step
    newpos = pos + (halfvel*step)
    newvel = halfvel + (.5*(-mu*newpos*step/norm(newpos)**3)) + .5*accel*step
    return newpos, newvel
def gravity_gradient(pos):
    pmag = np.linalg.norm(pos)
    return (mu / pmag**5) * (3 * np.outer(pos, pos) - pmag**2 * np.eye(3))
def indexInterp(arr, dblIndex,axis):
    if dblIndex>np.size(arr, axis=axis):
        raise ValueError("Index must be less than array axis max index.")
    if dblIndex<0:
        raise ValueError("Index must non-negative")
    i = int(dblIndex)
    d = dblIndex - i
    j=i+1
    return ((1-d) * np.take(arr, i, axis=axis)) + ((d) * np.take(arr, j, axis=axis))
def ECI2RIC(pos, vel, eps=1e-10):
    if norm(pos) < eps or norm(vel) < eps:
        return np.eye(3)
    radial = pos / norm(pos)
    cross = np.cross(pos, vel)
    cross = cross / norm(cross)
    intrack = np.cross(cross, radial)
    return np.vstack((radial, intrack, cross))

def printt(*args, file):
    print(*args)
    print(*args, file=file)