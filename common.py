import numpy as np
import math
import copy


class Test:
    def __init__(self, inp, outp):
        self.inputs = copy.deepcopy(inp)
        self.outputs = copy.deepcopy(outp)


# Афинные преобразования
# Поворот
def Rx(xyz, ang):
    angr = math.radians(ang)
    sin_ang = math.sin(angr)
    cos_ang = math.cos(angr)
    rx = np.array([[1, 0, 0, 0],
                   [0, cos_ang, sin_ang, 0],
                   [0, -sin_ang, cos_ang, 0],
                   [0, 0, 0, 1]])
    xyz = np.array(xyz)
    xyz = np.append(xyz, 1)
    res = np.dot(xyz, rx)
    return res[:3]

def Ry(xyz, ang):
    angr = math.radians(ang)
    sin_ang = math.sin(angr)
    cos_ang = math.cos(angr)
    ry = np.array([[cos_ang, 0, -sin_ang, 0],
                   [0, 1, 0, 0],
                   [sin_ang, 0, cos_ang, 0],
                   [0, 0, 0, 1]])
    xyz = np.array(xyz)
    xyz = np.append(xyz, 1)
    res = np.dot(xyz, ry)
    return res[:3]

def Rz(xyz, ang):
    angr = math.radians(ang)
    sin_ang = math.sin(angr)
    cos_ang = math.cos(angr)
    rz = np.array([[cos_ang, sin_ang, 0, 0],
                   [-sin_ang, cos_ang, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    xyz = np.array(xyz)
    xyz = np.append(xyz, 1)
    res = np.dot(xyz, rz)
    return res[:3]

# Перенос
def T(xyz, dx, dy, dz):
    t = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [dx, dy, dz, 1]])
    xyz = np.array(xyz)
    xyz = np.append(xyz, 1)
    res = np.dot(xyz, t)
    return res[:3]

# Отражение
def Myz(xyz):
    myz = np.array([[-1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    xyz = np.array(xyz)
    xyz = np.append(xyz, 1)
    res = np.dot(xyz, myz)
    return res[:3]


def get_distance_2d(p1, p2):
    res = math.sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]))
    return res


def get_distance_3d(p1, p2):
    res = math.sqrt(
        (p2[0] - p1[0]) * (p2[0] - p1[0]) +
        (p2[1] - p1[1]) * (p2[1] - p1[1]) +
        (p2[2] - p1[2]) * (p2[2] - p1[2]))
    return res
