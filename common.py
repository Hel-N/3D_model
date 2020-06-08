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
    rx = np.array([ [1, 0, 0, 0],
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
    res = math.sqrt((p2[0] - p1[0])*(p2[0] - p1[0]) + (p2[1] - p1[1])*(p2[1] - p1[1]))
    return res
def get_distance_3d(p1, p2):
    res = math.sqrt((p2[0] - p1[0])*(p2[0] - p1[0]) + (p2[1] - p1[1])*(p2[1] - p1[1]) + (p2[2] - p1[2])*(p2[2] - p1[2]))
    return res























# import copy
# import math
# import numpy as np
#
# class Pair:
#     first = None
#     second = None
#     def __init__(self, first, second):
#         self.first = first
#         self.second = second
#
#     def __lt__(self, other):
#         if self.first < other.first:
#             return True
#         elif self.first == other.first:
#             if self.second < other.second:
#                 return True
#         return False
#
# class Point:
#     def __init__(self, x = 0, y = 0, z = 0):
#         self.x = x
#         self.y = y
#         self.z = z
#
#
#
#
#
# class Line:
#     def __init__(self, a = Point(), b = Point(), line_length = 0):
#         self.a = copy.deepcopy(a)
#         self.b = copy.deepcopy(b)
#         self.length = line_length
#
# class Test:
# 	def __init__(self, inp, outp):
# 		self.inputs = copy.deepcopy(inp)
# 		self.outputs = copy.deepcopy(outp)
#
#
# def GetDistance(x1, y1, x2, y2):
#     res = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
#     return res
# def GetDistance2d(p1, p2):
#     res = math.sqrt((p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y))
#     return res
# def GetDistance3d(p1, p2):
#     res = math.sqrt((p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y) + (p2.z - p1.z)*(p2.z - p1.z))
#     return res
# def GetAngle(x1, y1, x2, y2):
#     res = 0.0
#     if (x1 == x2):
#         if (y1 == y2):
#             return res
#         if (y2 > y1):
#             return 1.0 * math.pi / 2
#         else:
#             return 3.0 * math.pi / 2
#
#     if (y1 == y2):
#         if (x2 > x1):
#             return  0.0
#         else:
#             return 1.0 * math.pi
#
#     res = math.atan((1.0 * abs(y2 - y1)) / (1.0 * abs(x2 - x1)))
#     if (x2 > x1 and y2 > y1):
#         return res
#     if (x2 < x1 and y2 > y1):
#         return math.pi - res
#     if (x2 < x1 and y2 < y1):
#         return math.pi + res
#     if (x2 > x1 and y2 < y1):
#         return 2 * math.pi - res
#
# def GetEquivPositiveAngle(angle):
#     while (angle < 0.0):
#         angle += 2.0 * math.pi
#     return angle
#
#
#  # перенос
# def T(x, y, z):
#     v =	[[ 1, 0, 0, 0],
# 		[0, 1, 0, 0],
# 		[0, 0, 1, 0],
# 		[x, y, z, 1 ]]
#     rx = np.array(v)
#     return rx
#
# def S(sx, sy, sz):
#     v = [[sx, 0, 0, 0],
#          [0, sy, 0, 0],
#          [0, 0, sz, 0],
#          [0, 0, 0, 1]]
#     rx = np.array(v)
#     return rx
#
# # повороты
# def Rx(a):
#     v =[ [1, 0, 0, 0],
# 		[0, math.cos(a), math.sin(a), 0],
# 		[0, -math.sin(a), math.cos(a), 0],
# 		[0, 0, 0, 1]]
#     rx = np.array(v)
#     return rx
#
# def Ry(a):
#     v =	[ [math.cos(a), 0, math.sin(a), 0],
# 		[0, 1, 0, 0],
# 		[-math.sin(a), 0, math.cos(a), 0],
# 		[0, 0, 0, 1 ]]
#     rx = np.array(v)
#     return rx
#
# def Rz(a):
#     v =	[ [math.cos(a), -math.sin(a), 0, 0],
# 		[math.sin(a), math.cos(a), 0, 0],
# 		[0, 0, 1, 0],
# 		[0, 0, 0, 1] ]
#     rx = np.array(v)
#     return rx
#
# def LookRotateMain(x, y, z):
#     p = np.array([x, y, z, 1])
#     res = np.dot(p, np.dot(Rx(-math.pi/7), Ry(-math.pi/6)))
#     return res
#
# def LookFront(x, y, z):
#     p = np.array([x, y, z, 1])
#     res = np.dot(p, np.dot(S(0.6, 0.6, 0.6), T(900, 250, 0)))
#     return res
#
# def LookOnTop(x, y, z):
#     p = np.array([x, y, z, 1])
#     res = np.dot(p, np.dot(np.dot(S(0.6, 0.6, 0.6), Rx(-math.pi/2)), T(900, 250, 0)))
#     return res
#
#
#
