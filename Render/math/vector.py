import numpy as np

def Rotation_Axis(ry:float, rx:float):
    ry1 = np.deg2rad(ry)
    rx1 = np.deg2rad(rx)
    z = np.cos(ry1) * np.cos(rx1)
    x = np.sin(ry1) * np.cos(rx1)
    y = np.sin(rx1)
    return np.array([x,y,z])

def Rotation_Matrix(ry:float, rx:float):
    v1 = Rotation_Axis(ry, rx)
    v2 = Rotation_Axis(ry, rx+90)
    v3 = Rotation_Axis(ry+90, 0)
    return np.array([v3,v2,v1])

