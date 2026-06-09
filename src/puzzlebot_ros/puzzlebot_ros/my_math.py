"""
my_math.py — Utility functions for PuzzleBot
"""

import math


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def euler_from_quaternion(x: float, y: float, z: float, w: float) -> tuple:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).
    Only yaw is relevant for a 2D robot.
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
