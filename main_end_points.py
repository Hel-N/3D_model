import copy
import math
import random
import time
import datetime
import os

from creature_end_points import Creature
from common import Test

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
import numpy as np

DBL_MAX = 1.79769e+308

fout_res = open("tmp_res.txt", "w")

# logout("log.txt")
# testout("test.txt")

# init_tests_filename = "point_init_tests.txt"
init_tests_filename = "tests.txt"

res_dir_str = "Res"
run_dist_finame_start = "res_dist_" # + текущее время
run_dist_finame_end = ".txt"

nnets_dir_str = "models"
nnet_finame_start = "point_model_" # + текущее время
nnet_finame_end = ".h5"

# NNet Config------------------------------------------------------
nnet_id = -1
nnet_name = ""
creature_name = "Hexapod"
NUM_HIDDEN_LAYERS = 2
NUM_HIDDEN_NEURONS = 100
ACT_FUNC = 'tanh'
END_ACT_FUNC = 'linear'
TOTAL_TESTS_NUMBER = 1000
CUR_TESTS_NUMBER = 100
EPOCH = 10
TRAINING_TYPE = 'rmsprop'

QGAMMA = 0.9 # Коэффициент доверия
TICK_COUNT = 10000
TRAIN_EPS = 0.001

ALL_DIST = 0
PREV_STEP_DIST = 1
CENTER_OF_BODY_Z = 2
REPEAT_ACTION = 3
used_reward = []
k_reward = {"CENTER_OF_BODY_Z": 0, "REPEAT_ACTION":0}

T = 500 #Период сохранения модели
RUN_TYPE = "RUN" #"TRAIN" # "RUN"

# For Model------------------------------------------------------------
creature_id = -1
creature_name = "Hexapod"
reward = 0.0
prev_dist = 0.0
cur_tick = 0

Q = []
prevQ = []
prev_action = -1
same_action_count = 0
first_step = True

inputs = []
prev_inputs = []

monster = None

# Tests---------------------------------------------------------------
tests = list(map(Test, []))
init_tests_count = 0

def CreatureInitialization():
    global monster

    # Constant----------------------------------------------------------------------------------------------------
    LEG_COUNT = 6
    robot_height = 12  # z
    legs_is_left = [True, True, True, False, False, False]
    coord_systems_transform = {"Rz": [-45, 0, 45, 135, 180, -135],
                              "dx": [25, 17.5, 10, 10, 17.5, 25],
                              "dy": [25, 27.5, 25, 17, 14.5, 17],
                              "dz": [robot_height, robot_height, robot_height, robot_height, robot_height,
                                     robot_height],
                              "Myz": [False, False, False, True, True, True]}
    start_foot_points = [[0, 12, -1.0 * robot_height], [0, 12, -1.0 * robot_height], [0, 12, -1.0 * robot_height],
                         [0, 12, -1.0 * robot_height], [0, 12, -1.0 * robot_height], [0, 12, -1.0 * robot_height]]

    leg_states_set = [
        # 0
        [[0, 10, -9.0],
         [0, 10, -12.0],
         [2.83, 12.83, -9.0],
         [2.83, 12.83, -12.0],
         [0, 12, -12.0]],
        # 1
        [[4, 10, -9.0],
         [4, 10, -12.0],
         [0, 10, -12.0],
         [0, 10, -9.0],
         [0, 12, -12.0]],
        # 2
        [[0, 10, -9.0],
         [0, 10, -12.0],
         [-2.83, 12.83, -12.0],
         [0, 12, -12.0]],
        # 3
        [[0, 10, -9.0],
         [0, 10, -12.0],
         [-2.83, 12.83, -12.0],
         [0, 12, -12.0]],
        # 4
        [[0, 10, -9.0],
         [0, 10, -12.0],
         [4, 10, -9.0],
         [4, 10, -12.0],
         [0, 12, -12.0]],
        # 5
        [[2.83, 12.83, -9.0],
         [2.83, 12.83, -12.0],
         [0, 10, -12.0],
         [0, 10, -9.0],
         [0, 12, -12.0]]
    ]
    # -------------------------------------------------------------------------------------------------------------

    monster = Creature(LEG_COUNT, robot_height, legs_is_left, coord_systems_transform, start_foot_points, leg_states_set)

def CreatureInitializationFromFile(filename):
    pass
    # global monster
    # global start_joints
    # with open(filename, "r") as fin:
    #     joints = []
    #     s = fin.readline()
    #     joint_count = int(fin.readline().strip())
    #     for i in range(joint_count):
    #         x, y, z = map(float, fin.readline().strip().split())
    #         joints.append(Point(x, y, z))
    #     start_joints = copy.deepcopy(joints)
    #
    #     lines = []
    #     s = fin.readline()
    #     line_count = int(fin.readline().strip())
    #     for i in range(line_count):
    #         idp1, idp2 = map(int, fin.readline().strip().split())
    #         lines.append(Pair(idp1, idp2))
    #
    #     mvlines = []
    #     s = fin.readline()
    #     mvlines_count = int(fin.readline().strip())
    #     for i in range(mvlines_count):
    #         line_id, joint_id = map(int, fin.readline().strip().split())
    #         mvlines.append(Pair(line_id, joint_id))
    #
    #
    #     s = fin.readline()
    #     fall_unit_angle = float(fin.readline().strip())
    #     s = fin.readline()
    #     turn_unit_angle = float(fin.readline().strip())
    #
    #     turnints = []
    #     s = fin.readline()
    #     turnint_count = int(fin.readline().strip())
    #     for i in range(turnint_count):
    #         min_ang, max_ang = map(float, fin.readline().strip().split())
    #         turnints.append(Pair(min_ang, max_ang))
    #
    #     mvstates = []
    #     s = fin.readline()
    #     mvstate_count = int(fin.readline().strip())
    #     for i in range(mvstate_count):
    #         st_num, st_count = map(int, fin.readline().strip().split())
    #         mvstates.append(Pair(st_num, st_count))
    #
    #     refs = []
    #     s = fin.readline()
    #     refs_count  = int(fin.readline().strip())
    #     for i in range(refs_count):
    #         refs_row = list(map(int, fin.readline().strip().split()))
    #         if len(refs_row) == 1:
    #             refs_row = []
    #         else:
    #             refs_row = refs_row[1:]
    #         refs.append(refs_row)
    #
    #     head_points = []
    #     s = fin.readline()
    #     head_points_count = int(fin.readline().strip())
    #     head_points = list(map(int, fin.readline().strip().split()))
    #
    #     monster.init_creature(joints, lines, mvlines, turnints, mvstates, refs, head_points)
    #     monster.fall_unit_angle = fall_unit_angle
    #     monster.turn_unit_angle = turn_unit_angle

def NNet():
    global monster, nnet, used_reward

    nnet = Sequential()
    nnet.add(Dense(NUM_HIDDEN_NEURONS, input_dim=3*4*monster.leg_count, activation=ACT_FUNC)) #'tanh'
    nnet.add(Dense(NUM_HIDDEN_NEURONS, activation=ACT_FUNC)) # 'tanh'
    nnet.add(Dense(monster.get_num_actions(), activation=END_ACT_FUNC)) # 'linear'

    # nnet.summary()
    # nnet.compile(optimizer='rmsprop', loss="mean_squared_error", metrics=["mean_squared_error"])

    nnet.compile(optimizer=TRAINING_TYPE, loss="mean_squared_error") # 'rmsprop'

    used_reward = [ALL_DIST]

def SaveNNet():
    global nnet, nnets_dir_str, nnet_finame_start, nnet_finame_end

    cur_time = datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')
    nnet.save(filepath=os.path.abspath(os.path.join(nnets_dir_str, nnet_finame_start, cur_time, nnet_finame_end)))

def SetInputs():
    global monster
    inp = []
    for i in range(len(monster.legs)):
        joint_pos = monster.legs[i].coxa_point
        for j in range(len(joint_pos)):
            inp.append(1.0*joint_pos[j])

        joint_pos = monster.legs[i].femur_point
        for j in range(len(joint_pos)):
            inp.append(1.0 * joint_pos[j])

        joint_pos = monster.legs[i].tibia_point
        for j in range(len(joint_pos)):
            inp.append(1.0 * joint_pos[j])

        joint_pos = monster.legs[i].end_point
        for j in range(len(joint_pos)):
            inp.append(1.0 * joint_pos[j])
    return copy.deepcopy(inp)

def AddTest(inp, outp):
    global tests, init_tests_count
    if len(tests) >= TOTAL_TESTS_NUMBER:
        del tests[init_tests_count]
    tests.append(Test(inp, outp))

def InitTests():
    global init_tests_count, monster, tests
    with open(init_tests_filename, "r") as fin:
        data = fin.readlines()

        actions_count = monster.get_num_actions()

        for i in range(0, len(data), 6):
            leg_id = int(data[i].strip())
            end_point = list(map(float, data[i+1].strip().split()))
            inps = list(map(float, data[i+2].strip().split()))
            all_dist = float(data[i+3].strip())

            state_id = monster.legs[leg_id].find_state(end_point)
            if (state_id == None):
                print("Error! Не найдено состояние для теста")

            for j in range(leg_id):
                state_id += len(monster.legs[j].states) - 1

            outps = [0.0 for i in range(actions_count)]
            outps[state_id] = all_dist

            AddTest(copy.deepcopy(inps), copy.deepcopy(outps))

        init_tests_count = len(tests)

def GetReward():
    global k_reward, used_reward, monster, prev_dist

    res = 0
    for i in range(len(used_reward)):
        rew = {
            ALL_DIST: math.fabs(monster.get_cur_delta_distance()),
            PREV_STEP_DIST: math.fabs(prev_dist - monster.get_cur_delta_distance()),
            CENTER_OF_BODY_Z: -k_reward["CENTER_OF_BODY_Z"] / max(1.0, monster.get_center_of_body()[2]),
            REPEAT_ACTION: -k_reward["REPEAT_ACTION"] / max(1.0, same_action_count)
        }[i]
        res += rew

    return res

def DoNextStep():
    global nnet, recovery_from_falling, monster, prev_dist, \
        prev_inputs, inputs, reward, cur_tick, first_step, \
        Q, prevQ, prev_action, tests, same_action_count


    prev_inputs = copy.deepcopy(inputs)
    inputs = SetInputs()
    action = -1
    reward = 0.0
    reward = GetReward()
    prev_dist = monster.get_cur_delta_distance()
    cur_tick += 1

    print("\nCur tick: {0}".format(cur_tick))
    print("All dist: {0}".format(prev_dist))

    if (not first_step):
        [Q] = nnet.predict(np.asarray([SetInputs()], dtype=np.float32))

        tmpQ = -DBL_MAX
        for i in range(len(Q)):
            if (tmpQ < Q[i]):
                tmpQ = Q[i]
                action = i

        Q[prev_action] = reward + QGAMMA*tmpQ

        if (RUN_TYPE == "TRAIN"):
            AddTest(prev_inputs, Q)
            epoch = EPOCH
            tests_pos = [i for i in range(init_tests_count)]
            for i in range(init_tests_count, min(CUR_TESTS_NUMBER, len(tests))):
                pos = max(0, random.randint(init_tests_count, len(tests) - 1))
                tests_pos.append(pos)

            cur_tests_in = np.asarray([copy.deepcopy(tests[tests_pos[i]].inputs) for i in range(len(tests_pos))], dtype=np.float32)
            cur_tests_out = np.asarray([copy.deepcopy(tests[tests_pos[i]].outputs) for i in range(len(tests_pos))], dtype=np.float32)
            nnet.fit(cur_tests_in, cur_tests_out, epochs=epoch, batch_size=min(1, CUR_TESTS_NUMBER), verbose=0)

    else:
        [Q] = nnet.predict(np.asarray([SetInputs()], dtype=np.float32))

        tmpQ = -DBL_MAX
        for i in range(len(Q)):
            if (tmpQ < Q[i]):
                tmpQ = Q[i]
                action = i
        first_step = False

    if (action != prev_action):
        monster.update_pos(action_num=action)
        same_action_count = 0
    else:
        same_action_count += 1

    if random.random() < 0.1:
        counter = 100
        flag_do = False
        for i in range(counter):
            action = random.randint(0, monster.get_num_actions() - 1)
            flag_do = True
            break
        if flag_do:
            if (action != prev_action):
                monster.update_pos(action_num=action)
                same_action_count = 0
            else:
                same_action_count += 1

    prev_action = action
    prevQ = copy.deepcopy(Q)

    #Сохранение модели
    # if (cur_tick % T == 0):
    #     SaveNNet()

def Timer(tick, coord_syst_lines, leg_lines, point_annotation):
    DoNextStep()
    Draw(tick, coord_syst_lines, leg_lines, point_annotation)


legs = []
coord_syst_lines = []
leg_lines = []
point_annotation = []


def InitDraw(ax):
    global monster, legs, coord_syst_lines, leg_lines, point_annotation

    for i in range(len(monster.legs)):
        coord_syst_lines.append([])
        cur_coord_syst = monster.legs[i].coord_system_for_plot(monster.cur_delta_distance_xyz)
        for j in range(len(cur_coord_syst)):
            color, start_p, end_p = cur_coord_syst[j]
            coord_syst_lines[i].append(ax.plot([start_p[0], end_p[0]],
                                            [start_p[1], end_p[1]],
                                            [start_p[2], end_p[2]], color)[0])

    for i in range(len(monster.legs)):
        plot_points = monster.legs[i].coord_for_plot(monster.cur_delta_distance_xyz)
        leg_lines.append(ax.plot(plot_points[0], plot_points[1], plot_points[2], marker = 'o', markerfacecolor = 'm',
                                 markeredgecolor = 'm', color='c')[0])

    for i in range(len(monster.legs)):
        end_p = monster.legs[i].transform_point_to_global(monster.legs[i].end_point, monster.cur_delta_distance_xyz)
        dx = 1
        dy = -3
        if (monster.legs[i].is_left):
            dy = 1
        point_annotation.append(ax.text3D(end_p[0] + dx, end_p[1] + dy, end_p[2], i, None))

    # Setting the axes properties
    ax.set_xlim3d([-20.0, 50.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-10.0, 50.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0.0, 30.0])
    ax.set_zlabel('Z')

def Draw(_tick, coord_syst_lines, leg_lines, point_annotation):
    global monster

    for i in range(len(monster.legs)):
        cur_coord_syst = monster.legs[i].coord_system_for_plot(monster.cur_delta_distance_xyz)
        for j in range(len(cur_coord_syst)):
            color, start_p, end_p = cur_coord_syst[j]
            coord_syst_lines[i][j].set_data([[start_p[0], end_p[0]],
                                             [start_p[1], end_p[1]]])
            coord_syst_lines[i][j].set_3d_properties([start_p[2], end_p[2]])

    for i in range(len(monster.legs)):
        plot_points = monster.legs[i].coord_for_plot(monster.cur_delta_distance_xyz)
        leg_lines[i].set_data(plot_points[0:2])
        leg_lines[i].set_3d_properties(plot_points[2])

    for i in range(len(monster.legs)):
        end_p = monster.legs[i].transform_point_to_global(monster.legs[i].end_point, monster.cur_delta_distance_xyz)
        dx = 1
        dy = -3
        if (monster.legs[i].is_left):
            dy = 1
        point_annotation[i].set_position([end_p[0] + dx, end_p[1] + dy])
        point_annotation[i].set_3d_properties(end_p[2])


    # Setting the axes properties
    cur_delta_distance_xyz = monster.cur_delta_distance_xyz
    ax.set_xlim3d([-20.0 + cur_delta_distance_xyz[0], 50.0 + cur_delta_distance_xyz[0]])
    ax.set_xlabel('X')

    ax.set_ylim3d([-10.0 + cur_delta_distance_xyz[1], 50.0 + cur_delta_distance_xyz[1]])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0.0 + cur_delta_distance_xyz[2], 30.0 + cur_delta_distance_xyz[2]])
    ax.set_zlabel('Z')

    return coord_syst_lines, leg_lines, point_annotation

if __name__ == "__main__":
    # crfilename = creatures_dir_str + creature_name + creature_finame_end
    # CreatureInitializationFromFile(crfilename)
    CreatureInitialization()

    # NNet()
    nnet = load_model('models/model_20200530-23-08-36.h5')

    # Создание папок
    directory = os.path.join(nnets_dir_str)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    InitTests()

    InitDraw(ax)

    line_animation = animation.FuncAnimation(fig, Timer, frames=TICK_COUNT,
                                             fargs=(coord_syst_lines, leg_lines, point_annotation),
                                             interval=25, blit=False)
    plt.show()




