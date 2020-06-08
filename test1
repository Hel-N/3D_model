import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import math
import copy
import time

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

# [x, y, z] - начальная и конечная точки
coord_system = [[[0, 0, 0], [3, 0, 0]],     # Ox
                [[0, 0, 0], [0, 3, 0]],     # Oy
                [[0, 0, 0], [0, 0, 3]]]     # Oz
robot_h = 12
L1 = 8.5
L2 = 12.5
coord_system_transform = {"Rz": [-45, 0, 45, 135, 180, -135],
                          "dx": [25, 17.5, 10, 10, 17.5, 25],
                          "dy": [25, 27.5, 25, 17, 14.5, 17],
                          "dz": [robot_h, robot_h, robot_h, robot_h, robot_h, robot_h],
                          "Myz": [False, False, False, True, True, True]}

legs_count = 6
lines = []
legs_lines = []
point_annotation = []
coord_annotation = []
cur_coord_systems = []
colors = ['g', 'b', 'r']

class Leg:
    def __init__(self):
        self.s0 = 0
        self.s1 = 0
        self.s2 = 0
        self.x0 = 0
        self.y0 = 0
        self.z0 = 14
        self.isLeft = False

    def attach(self, pov, col, nog, isLeft):
        pass

    def move(self, xm, ym, zm):
        pass

    def set(self, x2, y2, z2):
        pass

foot_points = [[0, 12, 0], [0, 12, 0], [0, 12, 0], [0, 12, 0], [0, 12, 0], [0, 12, 0]]
steps = [
    (5, [1.8, 12.6, 5]),
    (1, [5, 10, 5]),
    (3, [0, 10, 5]),
    (5, [1.8, 12.6, 0]),
    (1, [5, 10, 0]),
    (3, [0, 10, 0]),
    (4, [0, 10, 5]),
    (2, [0, 10, 5]),
    (0, [0, 10, 5]),
    (5, [0, 10, 0]),
    (3, [-2.7, 12.1, 0]),
    (1, [0, 10, 0]),
    (4, [0, 10, 0]),
    (2, [0, 10, 0]),
    (0, [0, 10, 0]),
    (0, [1.8, 12.6, 5]),
    (4, [5, 10, 5]),
    (2, [0, 10, 5]),
    (0, [1.8, 12.62, 0]),
    (4, [5, 10, 0]),
    (2, [0, 10, 0]),
    (1, [0, 10, 5]),
    (3, [0, 10, 5]),
    (5, [0, 10, 5]),
    (0, [0, 10, 0]),
    (2, [-2.7, 12.1, 0]),
    (4, [0, 10, 0]),
    (1, [0, 10, 0]),
    (3, [0, 10, 0]),
    (5, [0, 10, 0])
]

pause = [
    0.001,
    0.001,
    0.050,
    0.001,
    0.001,
    0.100,
    0.001,
    0.001,
    0.050,
    0.001,
    0.001,
    0.100,
    0.001,
    0.001,
    0.100,
    0.001,
    0.001,
    0.050,
    0.001,
    0.001,
    0.100,
    0.001,
    0.001,
    0.050,
    0.001,
    0.001,
    0.100,
    0.001,
    0.001,
    0.100
]

foot_cur_pos = []

def get_tibia_coords(leg_id, xm, ym, zm):
    global robot_h, L1, L2
    x0 = 0.0
    y0 = 0.0
    z0 = 0.0

    angs = [0, 0, 0]

    if (z0 >= zm):
        h = math.fabs(z0 - zm)
        s = math.fabs(ym - y0)
        m = h*h + s*s
        s1 = math.sqrt(m)
        u = (L1 * L1 + L2 * L2 - s1 * s1) / 2 / L1 / L2
        b = math.acos(u) * 180 / math.pi
        u = (s1 * s1 + L1 * L1 - L2 * L2) / 2 / s1 / L1
        v = math.acos(u) * 180 / math.pi
        u = h / s1
        p = math.acos(u) * 180 / math.pi
        a = 180 - v - p
        angs[1] = a
        angs[2] = b
    else:
        h = math.fabs(z0 - zm)
        s = math.fabs(ym - y0)
        m = h * h + s * s
        s1 = math.sqrt(m)
        u = (L1 * L1 + L2 * L2 - s1 * s1) / 2 / L1 / L2
        b = math.acos(u) * 180 / math.pi
        u = (s1 * s1 + L1 * L1 - L2 * L2) / 2 / s1 / L1
        v = math.acos(u) * 180 / math.pi
        u = h / s1
        p = math.asin(u) * 180 / math.pi # отличие asin
        a = 90 - v - p # отличие
        angs[1] = a
        angs[2] = b


    h = math.fabs(x0 - xm)
    s = math.fabs(y0 - ym)
    m = h * h + s * s
    s1 = math.sqrt(m)
    m = s / s1
    a = math.asin(m) * 180 / math.pi
    if (xm >= x0):
        angs[0] = a
    else:
        angs[0] = 180 - a

    x = 0.0
    y = 0.0
    z = robot_h
    dxy = 0.0

    #z
    if (angs[1] <= 90):
        z = z0 + L1*math.cos(math.radians(angs[1]))
        dxy = L1*math.sin(math.radians(angs[1]))
    else:
        z = z0 - L1 * math.sin(math.radians(angs[1]-90))
        dxy = L1 * math.cos(math.radians(angs[1]-90))

    # x y
    if (angs[0] > 90):
        x = -1.0 * dxy * math.sin(math.radians(angs[0] - 90))
        y = dxy * math.cos(math.radians(angs[0] - 90))
    else:
        x = dxy * math.cos(math.radians(angs[0]))
        y = dxy * math.sin(math.radians(angs[0]))

    return [x, y, z]

def update_pos(tick, lines, point_annotation, legs_lines):
    global steps, robot_h, legs_count, pause
    tick %= len(steps)
    (leg_id, point) = copy.deepcopy(steps[tick])

    xm, ym, zm = point

    if (coord_system_transform["Myz"][leg_id]):
        point = Myz(point)
    point = T(Rz(point, coord_system_transform["Rz"][leg_id]),
              coord_system_transform["dx"][leg_id],
              coord_system_transform["dy"][leg_id],
              coord_system_transform["dz"][leg_id])
    point[2] -= robot_h
    cur_data = [[], [], []]
    cur_data[0].append(point[0])
    cur_data[1].append(point[1])
    cur_data[2].append(point[2])

    lines[leg_id + legs_count*3].set_data(cur_data[0:2])
    lines[leg_id + legs_count*3].set_3d_properties(cur_data[2])

    point_annotation[leg_id].set_position([cur_data[0][0], cur_data[1][0]])
    point_annotation[leg_id].set_3d_properties(cur_data[2][0], None)

    x0 = 0.0
    y0 = 0.0
    z0 = 0.0
    leg_p = [[], [], []]
    leg_p[0] = [x0, y0, z0]
    leg_p[2] = [xm, ym, zm]
    leg_p[2][2] -= robot_h
    leg_p[1] = get_tibia_coords(k, xm, ym, zm - robot_h)

    cur_data = [[], [], []]
    for i in range(len(leg_p)):

        # if (leg_id == 2):
        #     coord_annotation[i].set_text(
        #         "({0:.1f};{1:.1f};{2:.1f})".format(leg_p[i][0], leg_p[i][1], leg_p[i][2]))

        if (coord_system_transform["Myz"][leg_id]):
            leg_p[i] = Myz(leg_p[i])
        leg_p[i] = T(Rz(leg_p[i], coord_system_transform["Rz"][leg_id]),
                     coord_system_transform["dx"][leg_id],
                     coord_system_transform["dy"][leg_id],
                     coord_system_transform["dz"][leg_id])
        for j in range(len(leg_p[i])):
            cur_data[j].append(leg_p[i][j])



    legs_lines[leg_id].set_data(cur_data[0:2])
    legs_lines[leg_id].set_3d_properties(cur_data[2])

    time.sleep(pause[tick] / 10000000)

    return lines, point_annotation, legs_lines

if __name__ == "__main__":
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    for k in range(legs_count):
        cur_coord_systems.append(copy.deepcopy(coord_system))
        for i in range(len(cur_coord_systems[k])):
            cur_data = [[], [], []]
            for j in range(len(cur_coord_systems[k][0])):
                if(coord_system_transform["Myz"][k]):
                    cur_coord_systems[k][i][j] = Myz(cur_coord_systems[k][i][j])
                cur_coord_systems[k][i][j] = T(Rz(cur_coord_systems[k][i][j], coord_system_transform["Rz"][k]),
                                               coord_system_transform["dx"][k], coord_system_transform["dy"][k], coord_system_transform["dz"][k])
                cur_data[0].append(cur_coord_systems[k][i][j][0])
                cur_data[1].append(cur_coord_systems[k][i][j][1])
                cur_data[2].append(cur_coord_systems[k][i][j][2])
            lines.append(ax.plot(cur_data[0], cur_data[1], cur_data[2], colors[i])[0])

    tmp = copy.deepcopy(foot_points)
    for k in range(legs_count):
        cur_data = [[], [], []]
        foot_cur_pos.append(foot_points[k])
        foot_cur_pos[k][2] -= robot_h
        if (coord_system_transform["Myz"][k]):
            foot_cur_pos[k] = Myz(foot_cur_pos[k])
        foot_cur_pos[k] = T(Rz(foot_cur_pos[k], coord_system_transform["Rz"][k]),
                          coord_system_transform["dx"][k], coord_system_transform["dy"][k], coord_system_transform["dz"][k])
        cur_data[0].append(foot_cur_pos[k][0])
        cur_data[1].append(foot_cur_pos[k][1])
        cur_data[2].append(foot_cur_pos[k][2])
        lines.append(ax.plot(cur_data[0], cur_data[1], cur_data[2], "mo")[0])
        point_annotation.append(ax.text3D(foot_cur_pos[k][0], foot_cur_pos[k][1], foot_cur_pos[k][2], k, None))
    foot_points = copy.deepcopy(tmp)

    for k in range(legs_count):
        x0 = 0.0
        y0 = 0.0
        z0 = 0.0
        leg_p = [[], [], []]
        leg_p[0] = [x0, y0, z0]
        leg_p[2] = copy.copy(foot_points[k])
        leg_p[2][2] -= robot_h
        leg_p[1] = get_tibia_coords(k, foot_points[k][0], foot_points[k][1], foot_points[k][2] - robot_h)

        cur_data = [[], [], []]
        for i in range(len(leg_p)):
            # if (k == 2):
            #     coord_annotation.append(ax.text3D(leg_p[i][0] + 0.1, leg_p[i][1] + i, leg_p[i][2],
            #                                       "({0:.1f};{1:.1f};{2:.1f})".format(leg_p[i][0], leg_p[i][1],
            #                                                                          leg_p[i][2]), None))

            if (coord_system_transform["Myz"][k]):
                leg_p[i] = Myz(leg_p[i])
            leg_p[i] = T(Rz(leg_p[i], coord_system_transform["Rz"][k]),
                          coord_system_transform["dx"][k],
                          coord_system_transform["dy"][k],
                          coord_system_transform["dz"][k])



            for j in range(len(leg_p[i])):
                cur_data[j].append(leg_p[i][j])
        legs_lines.append(ax.plot(cur_data[0], cur_data[1], cur_data[2], "c")[0])



    # Setting the axes properties
    ax.set_xlim3d([-10.0, 40.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-10.0, 40.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0.0, 20.0])
    ax.set_zlabel('Z')

    # Creating the Animation object
    line_animation = animation.FuncAnimation(fig, update_pos, len(steps), fargs=(lines, point_annotation, legs_lines), interval=50, blit=False)
    # line_animation = animation.FuncAnimation(fig, update_pos, len(steps), fargs=(lines, point_annotation, legs_lines, coord_annotation), interval=50, blit=False)

    plt.show()
