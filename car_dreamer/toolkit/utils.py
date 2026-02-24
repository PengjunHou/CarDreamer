import carla
import math


def _loc(a: carla.Actor) -> carla.Location:
    return a.get_transform().location


def _dist_m(a: carla.Actor, b: carla.Actor) -> float:
    la = _loc(a)
    lb = _loc(b)
    dx = la.x - lb.x
    dy = la.y - lb.y
    dz = la.z - lb.z
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))
