import numpy as np
import math
import copy
from common import Rx, Ry, Rz, T, Myz, get_distance_2d, get_distance_3d

DBL_MAX = 1.79769e+308

class Leg:
    L0 = 3      # coxa #cм
    L1 = 8.5    # femur #cм
    L2 = 12.5   # tibia #cм
    calc_eps = 0.01
    __id = None
    __is_left = None
    __coord_syst_transform = None # перемещение системы координат в точку сустава Coxa

    # Координаты сустава в системе координат ноги
    __coxa_point = None
    __femur_point = None
    __tibia_point = None
    __end_point = None
    __angs = None # [coxa_ang, femur_ang, tibia_ang] в градусах
    __cur_state = None
    # кол. состояний = кол. положений (кол. конечных точек) - возможные состояния (+ начальное состояние)
    __states = None # [coxa_st1, coxa_st2, ..., start_state]

    def __init__(self, id, is_left, coord_syst_transform, coxa_point, end_point, states_set):
        self.id = id
        self.is_left = is_left
        self.coord_syst_transform = coord_syst_transform
        self.coxa_point = coxa_point
        self.femur_point = [0.0, 0.0, 0.0]
        self.tibia_point = [0.0, 0.0, 0.0]
        self.end_point = [0.0, 0.0, 0.0]
        self.angs = [0.0, 0.0, 0.0]
        self.states = states_set
        self.__cur_state = None
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

    @property
    def states(self):
        return copy.deepcopy(self.__states)
    @states.setter
    def states(self, val):
        self.__states = copy.deepcopy(val)

    @property
    def cur_state(self):
        return copy.deepcopy(self.__cur_state)

    def num_joints(self):
        return len(self.angs)

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

        if (self.__angs[0] <= 0 or self.__angs[0] >= 180):
            raise Exception("Incorrect coxa angle")

        # fx fy
        if (self.angs[0] <= 90):
            self.__femur_point[0] = self.L0 * math.cos(math.radians(self.angs[0]))
            self.__femur_point[1] = self.L0 * math.sin(math.radians(self.angs[0]))
        else:
            self.__femur_point[0] = -self.L0 * math.sin(math.radians(self.angs[0] - 90))
            self.__femur_point[1] = self.L0 * math.cos(math.radians(self.angs[0] - 90))


        # Расчет углов поворота в суставах Femur и Tibia
        x0, y0, z0 = self.femur_point

        k = math.fabs(x0 - xm)
        h = math.fabs(z0 - zm)
        s = math.fabs(y0 - ym)
        ss = math.sqrt(k * k + h * h + s * s)
        u = (self.L1 * self.L1 + self.L2 * self.L2 - ss * ss) / 2 / self.L1 / self.L2
        b = math.acos(u) * 180 / math.pi
        u = (ss * ss + self.L1 * self.L1 - self.L2 * self.L2) / 2 / ss / self.L1
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

        if (self.__angs[1] <= 0 or self.__angs[1] >= 180):
            raise Exception("Incorrect femur angle")
        if (self.__angs[2] <= 0 or self.__angs[2] >= 180):
            raise Exception("Incorrect tibia angle")

        dxy = 0.0

        # tz
        x0, y0, z0 = self.coxa_point
        if (self.angs[1] <= 90):
            self.__tibia_point[2] = z0 + self.L1 * math.cos(math.radians(self.angs[1]))
            dxy = self.L1 * math.sin(math.radians(self.angs[1]))
        else:
            self.__tibia_point[2] = z0 - self.L1 * math.sin(math.radians(self.angs[1] - 90))
            dxy = self.L1 * math.cos(math.radians(self.angs[1] - 90))

        # tx ty
        if (self.angs[0] <= 90):
            self.__tibia_point[0] = (self.L0 + dxy) * math.cos(math.radians(self.angs[0]))
            self.__tibia_point[1] = (self.L0 + dxy) * math.sin(math.radians(self.angs[0]))
        else:
            self.__tibia_point[0] = -(self.L0 + dxy) * math.sin(math.radians(self.angs[0] - 90))
            self.__tibia_point[1] = (self.L0 + dxy) * math.cos(math.radians(self.angs[0] - 90))

        # Установка состоний
        state_id = self.find_state(self.end_point)
        self.__cur_state = state_id

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

    def rotate_joint(self, joint_id, ang):
        if (joint_id == 0): # coxa
            self.rotate_coxa(ang)
        elif (joint_id == 1): # femur
            self.rotate_femur(ang)
        elif (joint_id == 2): # tibia
            self.rotate_tibia(ang)
        state_id = self.find_state(self.end_point)
        self.__cur_state = state_id

    def rotate_coxa(self, ang):
        x0, y0, z0 = self.coxa_point

        # end_point
        h = math.fabs(x0 - self.__end_point[0])
        s = math.fabs(y0 - self.__end_point[1])
        ss = math.sqrt(h * h + s * s)
        if (ang <= 90):
            self.__end_point[0] = ss * math.cos(math.radians(ang))
            self.__end_point[1] = ss * math.sin(math.radians(ang))
        else:
            self.__end_point[0] = -ss * math.cos(math.radians(180 - ang))
            self.__end_point[1] = ss * math.sin(math.radians(180 - ang))

        # tibia_point
        h = math.fabs(x0 - self.__tibia_point[0])
        s = math.fabs(y0 - self.__tibia_point[1])
        ss = math.sqrt(h * h + s * s)
        if (ang <= 90):
            self.__tibia_point[0] = ss * math.cos(math.radians(ang))
            self.__tibia_point[1] = ss * math.sin(math.radians(ang))
        else:
            self.__tibia_point[0] = -ss * math.cos(math.radians(180 - ang))
            self.__tibia_point[1] = ss * math.sin(math.radians(180 - ang))

        # femur_point
        ss = self.L0
        if (ang <= 90):
            self.__femur_point[0] = ss * math.cos(math.radians(ang))
            self.__femur_point[1] = ss * math.sin(math.radians(ang))
        else:
            self.__femur_point[0] = -ss * math.cos(math.radians(180 - ang))
            self.__femur_point[1] = ss * math.sin(math.radians(180 - ang))

        self.__angs[0] = ang

    def rotate_femur(self, ang):
        x0, y0, z0 = self.femur_point

        # Рассчитываем угол v по старому положению ноги
        # v - угол между femur и отрезком Coxa-EndPoint
        k = math.fabs(x0 - self.__end_point[0])
        s = math.fabs(y0 - self.__end_point[1])
        h = math.fabs(z0 - self.__end_point[2])
        ss = math.sqrt(k * k + s * s + h * h)
        u = (ss * ss + self.L1 * self.L1 - self.L2 * self.L2) / 2 / ss / self.L1
        v = math.acos(u) * 180 / math.pi

        #end_point
        if((ang + v) <= 90):
            self.__end_point[2] = z0 + ss * math.sin(math.radians(90 - ang - v))
            dxy = ss * math.cos(math.radians(90 - ang - v))
        else:
            self.__end_point[2] = z0 - ss * math.cos(math.radians(180 - ang - v))
            dxy = ss * math.sin(math.radians(180 - ang - v))

        if (self.__angs[0] <= 90):
            self.__end_point[0] = (self.L0 + dxy) * math.cos(math.radians(self.__angs[0]))
            self.__end_point[1] = (self.L0 + dxy) * math.sin(math.radians(self.__angs[0]))
        else:
            self.__end_point[0] = -(self.L0 + dxy) * math.sin(math.radians(self.__angs[0] - 90))
            self.__end_point[1] = (self.L0 + dxy) * math.cos(math.radians(self.__angs[0] - 90))

        #tibia_point
        if (ang <= 90):
            self.__tibia_point[2] = z0 + self.L1 * math.sin(math.radians(90 - ang))
            dxy = self.L1 * math.cos(math.radians(90 - ang))
        else:
            self.__tibia_point[2] = z0 - self.L1 * math.cos(math.radians(180 - ang))
            dxy = self.L1 * math.sin(math.radians(180 - ang))

        if (self.__angs[0] <= 90):
            self.__tibia_point[0] = (self.L0 + dxy) * math.cos(math.radians(self.__angs[0]))
            self.__tibia_point[1] = (self.L0 + dxy) * math.sin(math.radians(self.__angs[0]))
        else:
            self.__tibia_point[0] = -(self.L0 + dxy) * math.sin(math.radians(self.__angs[0] - 90))
            self.__tibia_point[1] = (self.L0 + dxy) * math.cos(math.radians(self.__angs[0] - 90))

        self.__angs[1] = ang

    def rotate_tibia(self, ang):
        x0, y0, z0 = self.femur_point

        ss = math.sqrt(self.L1 * self.L1 + self.L2 * self.L2 - 2 * self.L1 * self.L2 * math.cos(math.radians(ang)))
        u = (ss * ss + self.L1 * self.L1 - self.L2 * self.L2) / 2 / ss / self.L1
        v = math.acos(u) * 180 / math.pi

        # end_point
        if ((self.__angs[1] + v) <= 90):
            self.__end_point[2] = z0 + ss * math.sin(math.radians(90 - self.__angs[1] - v))
            dxy = ss * math.cos(math.radians(90 - ang - v))
        else:
            self.__end_point[2] = z0 - ss * math.cos(math.radians(180 - self.__angs[1] - v))
            dxy = ss * math.sin(math.radians(180 - self.__angs[1] - v))

        if (self.__angs[0] <= 90):
            self.__end_point[0] = (self.L0 + dxy) * math.cos(math.radians(self.__angs[0]))
            self.__end_point[1] = (self.L0 + dxy) * math.sin(math.radians(self.__angs[0]))
        else:
            self.__end_point[0] = -(self.L0 + dxy) * math.sin(math.radians(self.__angs[0] - 90))
            self.__end_point[1] = (self.L0 + dxy) * math.cos(math.radians(self.__angs[0] - 90))

        self.__angs[2] = ang

    def find_state(self, end_point):
        for i in range(len(self.__states)):
            if (math.fabs(self.__states[i][0] - end_point[0]) < self.calc_eps and
                    math.fabs(self.__states[i][1] - end_point[1]) < self.calc_eps and
                    math.fabs(self.__states[i][2] - end_point[2]) < self.calc_eps):
                return i

        print("Warning! find_state end_point={0} Неизвестное состояние!".format(end_point))
        return None

    def transform_point_to_global(self, point, cur_pos = (0.0, 0.0, 0.0)):
        cur_dx, cur_dy, cur_dz = cur_pos
        if (self.__coord_syst_transform["Myz"]):
            point = Myz(point)
        point = T(T(Rz(point, self.__coord_syst_transform["Rz"]),
                         self.__coord_syst_transform["dx"],
                         self.__coord_syst_transform["dy"],
                         self.__coord_syst_transform["dz"]),
                         cur_dx, cur_dy, cur_dz)
        return point

    def transform_leg_coords(self, cur_pos = (0.0, 0.0, 0.0)):
        tr_coords = []
        tr_coords.append(self.coxa_point)
        tr_coords.append(self.femur_point)
        tr_coords.append(self.tibia_point)
        tr_coords.append(self.end_point)

        for i in range(len(tr_coords)):
            tr_coords[i] = self.transform_point_to_global(tr_coords[i], cur_pos)

        return tr_coords

    def coord_for_plot(self, cur_pos = (0.0, 0.0, 0.0)):
        plot_coord = [[], [], []] # x y z

        tr_coords = self.transform_leg_coords(cur_pos)

        for i in range(len(tr_coords)):
            for j in range(len(tr_coords[0])):
                plot_coord[j].append(tr_coords[i][j])

        return plot_coord

    def coord_system_for_plot(self, cur_pos=(0.0, 0.0, 0.0)):
        coord_system = [["g", [0, 0, 0], [3, 0, 0]],  # Ox
                        ["b", [0, 0, 0], [0, 3, 0]],  # Oy
                        ["r", [0, 0, 0], [0, 0, 3]]]  # Oz

        for i in range(len(coord_system)):
            coord_system[i][1] = self.transform_point_to_global(coord_system[i][1], cur_pos)
            coord_system[i][2] = self.transform_point_to_global(coord_system[i][2], cur_pos)
        return coord_system

    def servo_angs(self):
        sangs = copy.copy(self.angs)
        if(self.is_left):
            sangs[0] = 180 - self.angs[0]
            sangs[1] = 180 - self.angs[1]
            sangs[2] = 180 - self.angs[2]
        return sangs


# Размеры тела
#      * 8см *
#      |     |           | 7.5  |
# *_2.5|     |2.5_*      |      |  15 см
#      |     |           | 7.5  |
#      *     *

class Creature:
    leg_count = 6
    robot_height = 12
    ground_eps = 0.1
    legs = None
    start_body_center = None    # [x, y, z] - Центральная точка тела робота
    cur_body_center = None    # [x, y, z]
    cur_delta_distance_xyz = None # (dx, dy, dz)

    def __init__(self, leg_count, robot_height, legs_is_left, coord_systems_transform, start_foot_points, leg_states_set):
        self.leg_count = leg_count
        self.robot_height = robot_height

        self.legs = []
        for i in range(leg_count):
            coord_system_tr = {"Rz": coord_systems_transform["Rz"][i],
                               "dx": coord_systems_transform["dx"][i],
                               "dy": coord_systems_transform["dy"][i],
                               "dz": coord_systems_transform["dz"][i],
                               "Myz": coord_systems_transform["Myz"][i]}
            coxa_p = [0.0, 0.0, 0.0]
            end_p = start_foot_points[i]
            self.legs.append(Leg(i, legs_is_left[i], coord_system_tr, coxa_p, end_p, leg_states_set[i]))
        self.start_body_center = self.get_body_center()
        self.cur_body_center = self.get_body_center()
        self.cur_delta_distance_xyz = self.get_cur_delta_distance_xyz()

    def get_body_center(self):
        res = [(self.legs[1].coxa_point[0] + self.legs[4].coxa_point[0])/2.0,
               (self.legs[1].coxa_point[1] + self.legs[4].coxa_point[1])/2.0,
               (self.legs[1].coxa_point[2] + self.legs[4].coxa_point[2])/2.0]
        return res

    def get_cur_delta_distance(self):
        return get_distance_3d(self.start_body_center, self.cur_body_center)

    def get_cur_delta_distance_xyz(self):
        res = (self.cur_body_center[0] - self.start_body_center[0],
               self.cur_body_center[1] - self.start_body_center[1],
               self.cur_body_center[2] - self.start_body_center[2])
        return res

    def get_center_of_body(self):
        res = 1e6
        for i in range(len(self.legs)):
            end_point = self.legs[i].end_point
            if (end_point[2] < res):
                res = end_point[2]
        return [self.cur_body_center[0],
                self.cur_body_center[1],
                self.cur_body_center[2] - math.fabs(-self.robot_height - res)]


    def head_z(self):
        return self.robot_height

    def get_num_actions(self):
        res = 0
        for i in range(self.leg_count):
            res += len(self.legs[i].states) - 1 # Последнее состояние - начальное
        return res

    def get_action(self, action_num):
        leg_id = 0
        state_num = 0
        for i in range(self.leg_count):
            cur_st_count = len(self.legs[i].states) - 1 # Последнее состояние - начальное
            if (action_num - cur_st_count) < 0:
                state_num = action_num
                break
            else:
                action_num -= cur_st_count
                leg_id += 1

        return leg_id, state_num

    def can_do_action(self, action_num):
        leg_id, state_num = self.get_action(action_num)
        end_p = self.legs[leg_id].states[state_num]

        if (math.fabs(-self.robot_height - end_p[2]) > self.ground_eps):
            leg_up = []
            for i in range(self.leg_count):
                # if (i == leg_id):
                #     continue
                if (math.fabs(-self.robot_height - self.legs[i].end_point[2]) > self.ground_eps):
                    leg_up.append(i)

            if (leg_id in leg_up):
                return True

            if (len(leg_up) >= 3):
                return False

            if (len(leg_up) == 0 or len(leg_up) == 1):
                return True

            leg_up.append(leg_id)
            leg_up.sort()
            fl = False
            for i in range(len(leg_up)):
                if (abs(leg_up[i-1] - leg_up[i]) > 1 and
                        (leg_up[i-1] not in [0, 5]) and
                        (leg_up[i] not in [0, 5])):
                    fl = True

            return fl
        else:
            return True


    def move(self, leg_id, new_end_point):
        self.legs[leg_id].move(new_end_point)

    def rotate_joint(self, leg_id, joint_id, ang):
        self.legs[leg_id].rotate_joint(joint_id, ang)

    def do_action(self, action_num):
        leg_id, state_num = self.get_action(action_num)
        print("Info! do_action action_num = {}, leg_id = {}, state_num = {}".format(action_num, leg_id, state_num))
        new_end_point = self.legs[leg_id].states[state_num]
        self.legs[leg_id].move(new_end_point)

    def update_pos(self, leg_id = None, new_end_point=None, joint_id = None, ang = None, action_num = None):
        if (action_num != None):
            leg_id, _ = self.get_action(action_num)
            prev_end_p = self.legs[leg_id].transform_point_to_global(self.legs[leg_id].end_point, self.cur_delta_distance_xyz)
            self.do_action(action_num)
        elif(leg_id != None and new_end_point != None):
            prev_end_p = self.legs[leg_id].transform_point_to_global(self.legs[leg_id].end_point, self.cur_delta_distance_xyz)
            self.move(leg_id, new_end_point)
        elif (leg_id != None and joint_id != None and ang != None):
            prev_end_p = self.legs[leg_id].transform_point_to_global(self.legs[leg_id].end_point, self.cur_delta_distance_xyz)
            self.rotate_joint(leg_id, joint_id, ang)
        else:
            raise Exception('Update position error!')

        new_end_p = self.legs[leg_id].transform_point_to_global(self.legs[leg_id].end_point, self.cur_delta_distance_xyz)
        ground_point = self.legs[leg_id].transform_point_to_global([0, 0, -self.robot_height], self.cur_delta_distance_xyz)
        if (math.fabs(ground_point[2] - prev_end_p[2]) < self.ground_eps) and (math.fabs(ground_point[2] - new_end_p[2]) < self.ground_eps): # оттолкнулся
            delta_xyz = (prev_end_p[0] - new_end_p[0],
                         prev_end_p[1] - new_end_p[1],
                         prev_end_p[2] - new_end_p[2])
            self.cur_body_center = T(self.cur_body_center, delta_xyz[0], delta_xyz[1], delta_xyz[2])
            self.cur_delta_distance_xyz = self.get_cur_delta_distance_xyz()


