import numpy as np

def Rodrigues_Rotate(axis:np.ndarray, vector:np.ndarray, angle:float) -> np.ndarray:
    """旋转公式"""
    cos = np.cos(np.deg2rad(angle))
    sin = np.sin(np.deg2rad(angle))
    v1 = cos * vector
    v2 = (1 - cos) * np.dot(axis, vector) * axis
    v3 = np.cross(sin * axis, vector)
    return v1 + v2 + v3