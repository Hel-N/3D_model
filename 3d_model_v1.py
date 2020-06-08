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
coord_system = [("g", [0, 0, 0], [3, 0, 0]),     # Ox
                ("b", [0, 0, 0], [0, 3, 0]),     # Oy
                ("r", [0, 0, 0], [0, 0, 3])]     # Oz
robot_height = 12 # z

L0 = 3
L1 = 8.5
L2 = 12.5
LEG_COUNT = 6

class Leg:
    __id = None
    __is_left = None
    __coord_syst_transform = None # перемещение системы координат в точку сустава Coxa
    # Координаты сустава в системе координат ноги
    __coxa_point = None
    __femur_point = None
    __tibia_point = None
    __end_point = None
    __angs = None # [coxa_ang, femur_ang, tibia_ang] в градусах

    def __init__(self, id, is_left, coord_syst_transform, coxa_point, end_point):
        self.id = id
        self.is_left = is_left
        self.coord_syst_transform = coord_syst_transform
        self.coxa_point = coxa_point
        self.femur_point = [0.0, 0.0, 0.0]
        self.tibia_point = [0.0, 0.0, 0.0]
        self.end_point = [0.0, 0.0, 0.0]
        self.angs = [0.0, 0.0, 0.0]  # [coxa_ang, femur_ang, tibia_ang] в градусах
        self.move(end_point)

    @property
    def id(self):
        return self.__id
    @id.setter
    def id(self, val):
        self.__id = val

    @property
    def is_left(self):
        return self.__is_left
    @is_left.setter
    def is_left(self, val):
        self.__is_left = val

    @property
    def coord_syst_transform(self):
        return copy.deepcopy(self.__coord_syst_transform)
    @coord_syst_transform.setter
    def coord_syst_transform(self, val):
        self.__coord_syst_transform = copy.deepcopy(val)

    @property
    def coxa_point(self):
        return copy.deepcopy(self.__coxa_point)
    @coxa_point.setter
    def coxa_point(self, val):
        self.__coxa_point = copy.deepcopy(val)

    @property
    def femur_point(self):
        return copy.deepcopy(self.__femur_point)
    @femur_point.setter
    def femur_point(self, val):
        self.__femur_point = copy.deepcopy(val)

    @property
    def tibia_point(self):
        return copy.deepcopy(self.__tibia_point)
    @tibia_point.setter
    def tibia_point(self, val):
        self.__tibia_point= copy.deepcopy(val)

    @property
    def end_point(self):
        return copy.deepcopy(self.__end_point)
    @end_point.setter
    def end_point(self, val):
        self.__end_point = copy.deepcopy(val)

    @property
    def angs(self):
        return copy.deepcopy(self.__angs)
    @angs.setter
    def angs(self, val):
        self.__angs= copy.deepcopy(val)


    def move(self, new_end_point):
        self.end_point = new_end_point
        xm, ym, zm = new_end_point

        # Расчет угла поворота в суставе Coxa
        x0, y0, z0 = self.coxa_point

        h = math.fabs(x0 - xm)
        s = math.fabs(y0 - ym)
        ss = math.sqrt(h * h + s * s)
        m = s / ss
        a = math.asin(m) * 180 / math.pi
        if (xm >= x0):
            self.__angs[0] = a
        else:
            self.__angs[0] = 180 - a

        # fx fy
        if (self.angs[0] <= 90):
            self.__femur_point[0] = L0 * math.cos(math.radians(self.angs[0]))
            self.__femur_point[1] = L0 * math.sin(math.radians(self.angs[0]))
        else:
            self.__femur_point[0] = -L0 * math.sin(math.radians(self.angs[0] - 90))
            self.__femur_point[1] = L0 * math.cos(math.radians(self.angs[0] - 90))


        # Расчет углов поворота в суставах Femur и Tibia
        x0, y0, z0 = self.femur_point

        h = math.fabs(z0 - zm)
        s = math.fabs(y0 - ym)
        ss = math.sqrt(h * h + s * s)
        u = (L1 * L1 + L2 * L2 - ss * ss) / 2 / L1 / L2
        b = math.acos(u) * 180 / math.pi
        u = (ss * ss + L1 * L1 - L2 * L2) / 2 / ss / L1
        v = math.acos(u) * 180 / math.pi
        u = h / ss
        if (z0 >= zm):
            p = math.acos(u) * 180 / math.pi
            a = 180 - v - p
            self.__angs[1] = a
            self.__angs[2] = b
        else:
            p = math.asin(u) * 180 / math.pi  # отличие asin
            a = 90 - v - p  # отличие
            self.__angs[1] = a
            self.__angs[2] = b

        dxy = 0.0

        # tz
        x0, y0, z0 = self.coxa_point
        if (self.angs[1] <= 90):
            self.__tibia_point[2] = z0 + L1 * math.cos(math.radians(self.angs[1]))
            dxy = L1 * math.sin(math.radians(self.angs[1]))
        else:
            self.__tibia_point[2] = z0 - L1 * math.sin(math.radians(self.angs[1] - 90))
            dxy = L1 * math.cos(math.radians(self.angs[1] - 90))

        # tx ty
        if (self.angs[0] <= 90):
            self.__tibia_point[0] = (L0 + dxy) * math.cos(math.radians(self.angs[0]))
            self.__tibia_point[1] = (L0 + dxy) * math.sin(math.radians(self.angs[0]))
        else:
            self.__tibia_point[0] = -(L0 + dxy) * math.sin(math.radians(self.angs[0] - 90))
            self.__tibia_point[1] = (L0 + dxy) * math.cos(math.radians(self.angs[0] - 90))

        # tmp1 = math.sqrt((self.coxa_point[0]-self.femur_point[0])*(self.coxa_point[0]-self.femur_point[0]) +
        #                  (self.coxa_point[1]-self.femur_point[1])*(self.coxa_point[1]-self.femur_point[1]) +
        #                  (self.coxa_point[2]-self.femur_point[2])*(self.coxa_point[2]-self.femur_point[2]))
        # tmp2 = math.sqrt((self.femur_point[0] - self.tibia_point[0]) * (self.femur_point[0] - self.tibia_point[0]) +
        #                  (self.femur_point[1] - self.tibia_point[1]) * (self.femur_point[1] - self.tibia_point[1]) +
        #                  (self.femur_point[2] - self.tibia_point[2]) * (self.femur_point[2] - self.tibia_point[2]))
        # tmp3 = math.sqrt((self.tibia_point[0] - self.end_point[0]) * (self.tibia_point[0] - self.end_point[0]) +
        #                  (self.tibia_point[1] - self.end_point[1]) * (self.tibia_point[1] - self.end_point[1]) +
        #                  (self.tibia_point[2] - self.end_point[2]) * (self.tibia_point[2] - self.end_point[2]))
        # print(tmp1)
        # print(tmp2)
        # print(tmp3)
        # print("")

        return None

    def transform_point(self, point):
        if (self.__coord_syst_transform["Myz"]):
            point = Myz(point)
        point = T(Rz(point, self.__coord_syst_transform["Rz"]),
                         self.__coord_syst_transform["dx"],
                         self.__coord_syst_transform["dy"],
                         self.__coord_syst_transform["dz"])
        return point

    def transform_leg_coords(self):
        tr_coords = []
        tr_coords.append(self.coxa_point)
        tr_coords.append(self.femur_point)
        tr_coords.append(self.tibia_point)
        tr_coords.append(self.end_point)

        for i in range(len(tr_coords)):
            tr_coords[i] = self.transform_point(tr_coords[i])

        return tr_coords

    def coord_for_plot(self):
        plot_coord = [[], [], []]

        tr_coords = self.transform_leg_coords()

        for i in range(len(tr_coords)):
            for j in range(len(tr_coords[0])):
                plot_coord[j].append(tr_coords[i][j])

        # tmp1 = math.sqrt((plot_coord[0][0]-plot_coord[0][1])*(plot_coord[0][0]-plot_coord[0][1]) +
        #                  (plot_coord[1][0]-plot_coord[1][1])*(plot_coord[1][0]-plot_coord[1][1]) +
        #                  (plot_coord[2][0]-plot_coord[2][1])*(plot_coord[2][0]-plot_coord[2][1]))
        # tmp2 = math.sqrt((plot_coord[0][1]-plot_coord[0][2])*(plot_coord[0][1]-plot_coord[0][2]) +
        #                  (plot_coord[1][1]-plot_coord[1][2])*(plot_coord[1][1]-plot_coord[1][2]) +
        #                  (plot_coord[2][1]-plot_coord[2][2])*(plot_coord[2][1]-plot_coord[2][2]))
        # tmp3 = math.sqrt((plot_coord[0][2]-plot_coord[0][3])*(plot_coord[0][2]-plot_coord[0][3]) +
        #                  (plot_coord[1][2]-plot_coord[1][3])*(plot_coord[1][2]-plot_coord[1][3]) +
        #                  (plot_coord[2][2]-plot_coord[2][3])*(plot_coord[2][2]-plot_coord[2][3]))
        # print(tmp1)
        # print(tmp2)
        # print(tmp3)
        # print("")

        return plot_coord

leg_is_left = [True, True, True, False, False, False]
coord_system_transform = {"Rz": [-45, 0, 45, 135, 180, -135],
                          "dx": [25, 17.5, 10, 10, 17.5, 25],
                          "dy": [25, 27.5, 25, 17, 14.5, 17],
                          "dz": [robot_height, robot_height, robot_height, robot_height, robot_height, robot_height],
                          "Myz": [False, False, False, True, True, True]}
start_foot_points = [[0, 12, -1.0*robot_height], [0, 12, -1.0*robot_height], [0, 12, -1.0*robot_height],
                     [0, 12, -1.0*robot_height], [0, 12, -1.0*robot_height], [0, 12, -1.0*robot_height]]
# steps = [
#     (5, [1.8, 12.6, 5-1.0*robot_height]),
#     (1, [5, 10, 5-1.0*robot_height]),
#     (3, [0, 10, 5-1.0*robot_height]),
#
#     (5, [1.8, 12.6, -1.0*robot_height]),
#     (1, [5, 10, -1.0*robot_height]),
#     (3, [0, 10, -1.0*robot_height]),
#
#     (4, [0, 10, 5-1.0*robot_height]),
#     (2, [0, 10, 5-1.0*robot_height]),
#     (0, [0, 10, 5-1.0*robot_height]),
#
#     (5, [0, 10, -1.0*robot_height]),
#     (3, [-2.7, 12.1, -1.0*robot_height]),
#     (1, [0, 10, -1.0*robot_height]),
#
#     (4, [0, 10, -1.0*robot_height]),
#     (2, [0, 10, -1.0*robot_height]),
#     (0, [0, 10, -1.0*robot_height]),
#
#     (0, [1.8, 12.6, 5-1.0*robot_height]),
#     (4, [5, 10, 5-1.0*robot_height]),
#     (2, [0, 10, 5-1.0*robot_height]),
#
#     (0, [1.8, 12.62, -1.0*robot_height]),
#     (4, [5, 10, -1.0*robot_height]),
#     (2, [0, 10, -1.0*robot_height]),
#
#     (1, [0, 10, 5-1.0*robot_height]),
#     (3, [0, 10, 5-1.0*robot_height]),
#     (5, [0, 10, 5-1.0*robot_height]),
#
#     (0, [0, 10, -1.0*robot_height]),
#     (2, [-2.7, 12.1, -1.0*robot_height]),
#     (4, [0, 10, -1.0*robot_height]),
#
#     (1, [0, 10, -1.0*robot_height]),
#     (3, [0, 10, -1.0*robot_height]),
#     (5, [0, 10, -1.0*robot_height])
# ]
steps = [
    (5, [2.83, 12.83, 3-1.0*robot_height]),
    (1, [4, 10, 3-1.0*robot_height]),
    (3, [0, 10, 3-1.0*robot_height]),

    (5, [2.83, 12.83, -1.0*robot_height]),
    (1, [4, 10, -1.0*robot_height]),
    (3, [0, 10, -1.0*robot_height]),

    (4, [0, 10, 3-1.0*robot_height]),
    (2, [0, 10, 3-1.0*robot_height]),
    (0, [0, 10, 3-1.0*robot_height]),

    (5, [0, 10, -1.0*robot_height]),
    (3, [-2.83, 12.83, -1.0*robot_height]),
    (1, [0, 10, -1.0*robot_height]),

    (4, [0, 10, -1.0*robot_height]),
    (2, [0, 10, -1.0*robot_height]),
    (0, [0, 10, -1.0*robot_height]),

    (0, [2.83, 12.83, 3-1.0*robot_height]),
    (4, [4, 10, 3-1.0*robot_height]),
    (2, [0, 10, 3-1.0*robot_height]),

    (0, [2.83, 12.83, -1.0*robot_height]),
    (4, [4, 10, -1.0*robot_height]),
    (2, [0, 10, -1.0*robot_height]),

    (1, [0, 10, 3-1.0*robot_height]),
    (3, [0, 10, 3-1.0*robot_height]),
    (5, [0, 10, 3-1.0*robot_height]),

    (0, [0, 10, -1.0*robot_height]),
    (2, [-2.83, 12.83, -1.0*robot_height]),
    (4, [0, 10, -1.0*robot_height]),

    (1, [0, 10, -1.0*robot_height]),
    (3, [0, 10, -1.0*robot_height]),
    (5, [0, 10, -1.0*robot_height])
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

legs = []
coord_syst_lines = []
leg_lines = []
point_annotation = []

cur_delta_dist_x = 0.0

def update_pos(tick, coord_syst_lines, leg_lines, point_annotation):
    global legs, coord_system, steps, pause

    tick %= len(steps)
    (leg_id, new_end_p) = copy.deepcopy(steps[tick])
    legs[leg_id].move(new_end_p)

    for i in range(len(legs)):
        for j in range(len(coord_system)):
            color, start_p, end_p = coord_system[j]
            start_p = legs[i].transform_point(copy.copy(start_p))
            end_p = legs[i].transform_point(copy.copy(end_p))
            coord_syst_lines[i][j].set_data([[start_p[0], end_p[0]],
                                             [start_p[1], end_p[1]]])
            coord_syst_lines[i][j].set_3d_properties([start_p[2], end_p[2]])

    for i in range(len(legs)):
        plot_points = legs[i].coord_for_plot()
        leg_lines[i].set_data(plot_points[0:2])
        leg_lines[i].set_3d_properties(plot_points[2])

    for i in range(len(legs)):
        end_p = legs[i].transform_point(legs[i].end_point)
        dx = 1
        dy = -3
        if (legs[i].is_left):
            dy = 1
        point_annotation[i].set_position([end_p[0] + dx, end_p[1] + dy])
        point_annotation[i].set_3d_properties(end_p[2])


    time.sleep(pause[tick] / 10000000)

    return coord_syst_lines, leg_lines, point_annotation



if __name__ == "__main__":

    for i in range(LEG_COUNT):
        coord_system_tr = {"Rz": coord_system_transform["Rz"][i],
                           "dx": coord_system_transform["dx"][i],
                           "dy": coord_system_transform["dy"][i],
                           "dz": coord_system_transform["dz"][i],
                           "Myz": coord_system_transform["Myz"][i]}
        coxa_p = [0, 0, 0]
        end_p = start_foot_points[i]
        legs.append(Leg(i, leg_is_left[i], coord_system_tr, coxa_p, end_p))


    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    for i in range(len(legs)):
        coord_syst_lines.append([])
        for j in range(len(coord_system)):
            color, start_p, end_p = coord_system[j]
            start_p = legs[i].transform_point(copy.copy(start_p))
            end_p = legs[i].transform_point(copy.copy(end_p))
            coord_syst_lines[i].append(ax.plot([start_p[0], end_p[0]],
                                            [start_p[1], end_p[1]],
                                            [start_p[2], end_p[2]], color)[0])

    for i in range(len(legs)):
        plot_points = legs[i].coord_for_plot()
        leg_lines.append(ax.plot(plot_points[0], plot_points[1], plot_points[2], marker = 'o', markerfacecolor = 'm', markeredgecolor = 'm',
                                 color='c')[0])

    for i in range(len(legs)):
        end_p = legs[i].transform_point(legs[i].end_point)
        dx = 1
        dy = -3
        if (legs[i].is_left):
            dy = 1
        point_annotation.append(ax.text3D(end_p[0] + dx, end_p[1] + dy, end_p[2], i, None))


    # Setting the axes properties
    ax.set_xlim3d([-10.0, 50.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-10.0, 50.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0.0, 50.0])
    ax.set_zlabel('Z')

    line_animation = animation.FuncAnimation(fig, update_pos, len(steps), fargs=(coord_syst_lines, leg_lines, point_annotation),
                                             interval=50, blit=False)


    plt.show()

