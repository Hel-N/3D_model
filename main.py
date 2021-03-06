import copy
import math
import random
import sys
import time
import os
import datetime

from creature import Creature
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
init_tests_filename = "tests1.txt"

res_dir_str = "Res_ang"
run_dist_finame_start = "res_dist_" # + текущее время
run_dist_finame_end = ".txt"

creature_dir_str = "Creatures"
creature_finame_end = ".txt"

nnets_dir_str = "models_angs"
# nnet_finame_start = "ang_model_" # + текущее время
nnet_finame_end = ".h5"

# NNet Config------------------------------------------------------
nnet_id = -1
nnet_name = ""
creature_name = ""
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

T = 100 #Период сохранения модели
RUN_TYPE = "TRAIN" # "RUN"

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
    # [x, y, z] - начальная и конечная точки
    # coord_system = [("g", [0, 0, 0], [3, 0, 0]),  # Ox
    #                 ("b", [0, 0, 0], [0, 3, 0]),  # Oy
    #                 ("r", [0, 0, 0], [0, 0, 3])]  # Oz
    legs_is_left = [True, True, True, False, False, False]
    coord_systems_transform = {"Rz": [-45, 0, 45, 135, 180, -135],
                              "dx": [25, 17.5, 10, 10, 17.5, 25],
                              "dy": [25, 27.5, 25, 17, 14.5, 17],
                              "dz": [robot_height, robot_height, robot_height, robot_height, robot_height,
                                     robot_height],
                              "Myz": [False, False, False, True, True, True]}
    start_foot_points = [[0, 12, -1.0 * robot_height], [0, 12, -1.0 * robot_height], [0, 12, -1.0 * robot_height],
                         [0, 12, -1.0 * robot_height], [0, 12, -1.0 * robot_height], [0, 12, -1.0 * robot_height]]
    # # последнее состояние - начальное
    # leg_states_set = [
    #     # 0
    #     [[90.0, 90.0, 77.561064239824, 77.561064239824, 90.0, 90.0],
    #      [65.85352803393266, 87.22937300022821, 66.96468639421255, 87.16688822780962, 87.22937300022821, 86.69912724436122],
    #      [62.38491044515201, 80.38314334918212, 76.53479387112577, 93.64737729881365, 80.38314334918212, 89.05626213516145]],
    #     # 1
    #     [[68.19859051364821, 68.19859051364821, 90.0, 90.0, 90.0, 90.0],
    #      [65.76946313551514, 87.06735350343983, 87.22937300022821, 65.85352803393266, 87.22937300022821, 86.69912724436122],
    #      [63.30914637883293, 81.21623613932147, 80.38314334918212, 62.38491044515201, 80.38314334918212, 89.05626213516145]],
    #     # 2
    #     [[90.0, 90.0, 90.0, 90.0, 102.438935760176, 90.0],
    #      [65.85352803393266, 87.22937300022821, 65.85352803393266, 87.22937300022821, 87.16688822780962, 86.69912724436122],
    #      [62.38491044515201, 80.38314334918212, 62.38491044515201, 80.38314334918212, 93.64737729881365, 89.05626213516145]],
    #     # 3
    #     [[90.0, 90.0, 102.438935760176, 90.0, 90.0, 90.0],
    #      [65.85352803393266, 87.22937300022821, 87.16688822780962, 65.85352803393266, 87.22937300022821, 86.69912724436122],
    #      [62.38491044515201, 80.38314334918212, 93.64737729881365, 62.38491044515201, 80.38314334918212, 89.05626213516145]],
    #     # 4
    #     [[90.0, 90.0, 68.19859051364821, 68.19859051364821, 90.0, 90.0],
    #      [65.85352803393266, 87.22937300022821, 65.76946313551514, 87.06735350343983, 87.22937300022821, 86.69912724436122],
    #      [62.38491044515201, 80.38314334918212, 63.30914637883293, 81.21623613932147, 80.38314334918212, 89.05626213516145]],
    #     # 5
    #     [[77.561064239824, 77.561064239824, 90.0, 90.0, 90.0, 90.0],
    #      [66.96468639421255, 87.16688822780962, 87.22937300022821, 65.85352803393266, 87.22937300022821, 86.69912724436122],
    #      [76.53479387112577, 93.64737729881365, 80.38314334918212, 62.38491044515201, 80.38314334918212, 89.05626213516145]]
    #      ]
    # последнее состояние - начальное
    leg_states_set = [
        # 0
        [[90.0, 77.561064239824, 90.0],
         [65.85352803393266, 87.22937300022821, 67.2527731592374, 87.36368870410308, 86.69912724436122],
         [62.38491044515201, 80.38314334918212, 77.85347317045873, 94.93687188154904, 89.05626213516145]],
        # 1
        [[68.19859051364821, 90.0, 90.0],
         [65.68443629105134, 86.76633604129218, 87.22937300022821,  65.85352803393266, 86.69912724436122],
         [65.79605864031062, 83.48200916190999, 80.38314334918212,  62.38491044515201, 89.05626213516145]],
        # 2
        [[90.0, 102.438935760176, 90.0],
         [65.85352803393266, 87.22937300022821, 87.36368870410308, 86.69912724436122],
         [62.38491044515201, 80.38314334918212, 94.93687188154904, 89.05626213516145]],
        # 3
        [[90.0, 102.438935760176, 90.0],
         [65.85352803393266, 87.22937300022821, 87.36368870410308, 86.69912724436122],
         [62.38491044515201, 80.38314334918212, 94.93687188154904, 89.05626213516145]],
        # 4
        [[90.0, 68.19859051364821, 90.0],
         [65.85352803393266, 87.22937300022821, 65.68443629105134,  86.76633604129218, 86.69912724436122],
         [62.38491044515201, 80.38314334918212, 65.79605864031062,  83.48200916190999, 89.05626213516145]],
        # 5
        [[77.561064239824, 90.0, 90.0],
         [67.2527731592374, 87.36368870410308, 87.22937300022821,  65.85352803393266, 86.69912724436122],
         [77.85347317045873, 94.93687188154904, 80.38314334918212,  62.38491044515201, 89.05626213516145]]
         ]
    # -------------------------------------------------------------------------------------------------------------

    monster = Creature(LEG_COUNT, robot_height, legs_is_left, coord_systems_transform, start_foot_points, leg_states_set)

def CreatureInitializationFromFile(filename):
    global monster, used_reward, k_reward

    with open(filename, "r") as fin:
        leg_count_str = fin.readline().strip().split('=')
        leg_count = int(leg_count_str[1].strip())

        robot_height_str = fin.readline().strip().split('=')
        robot_height = int(robot_height_str[1].strip())

        legs_is_left_str = fin.readline().strip().split(':')
        legs_is_left = list(map(bool, legs_is_left_str[1].split()))

        coord_systems_transform = {}
        fin.readline()
        rz_str = fin.readline().strip().split(':')
        rz = list(map(float, rz_str[1].split()))
        coord_systems_transform["Rz"] = rz

        dx_str = fin.readline().strip().split(':')
        dx = list(map(float, dx_str[1].split()))
        coord_systems_transform["dx"] = dx

        dy_str = fin.readline().strip().split(':')
        dy = list(map(float, dy_str[1].split()))
        coord_systems_transform["dy"] = dy

        dz_str = fin.readline().strip().split(':')
        dz = list(map(float, dz_str[1].split()))
        coord_systems_transform["dz"] = dz

        myz_str = fin.readline().strip().split(':')
        myz = list(map(bool, myz_str[1].split()))
        coord_systems_transform["Myz"] = myz

        start_foot_points = []
        fin.readline()
        for i in range(leg_count):
            cur_sp = list(map(float, fin.readline().strip().split()))
            start_foot_points.append(copy.deepcopy(cur_sp))

        leg_states_set = []
        fin.readline()
        for i in range(leg_count):
            leg_id = list(map(int, fin.readline().split()))
            cur_states_set = []
            for j in range(3):
                cur_p = list(map(float, fin.readline().strip().split()))
                cur_states_set.append(copy.deepcopy(cur_p))
            leg_states_set.append(copy.deepcopy(cur_states_set))

        # Reward
        fin.readline()

        all_dist_str = fin.readline().strip().split('=')
        fl = int(all_dist_str[1].strip())
        if (fl == 1):
            used_reward.append(ALL_DIST)

        prev_step_dist_str = fin.readline().strip().split('=')
        fl = int(prev_step_dist_str[1].strip())
        if (fl == 1):
            used_reward.append(PREV_STEP_DIST)

        center_of_body_str = fin.readline().strip().split('=')
        fl = int(center_of_body_str[1].strip())
        if (fl == 1):
            used_reward.append(CENTER_OF_BODY_Z)

        repeat_action_str = fin.readline().strip().split('=')
        fl = int(repeat_action_str[1].strip())
        if (fl == 1):
            used_reward.append(REPEAT_ACTION)

        k_cob_str = fin.readline().strip().split('=')
        k = float(k_cob_str[1].strip())
        k_reward["CENTER_OF_BODY_Z"] = k

        k_ra_str = fin.readline().strip().split('=')
        k = float(k_ra_str[1].strip())
        k_reward["REPEAT_ACTION"] = k

        monster = Creature(leg_count, robot_height, legs_is_left, coord_systems_transform, start_foot_points, leg_states_set)
def NNet():
    global monster, nnet
    # model.add(Dense(number_of_neurons, input_dim=number_of_cols_in_input, activtion=some_activation_function)).

    nnet = Sequential()
    nnet.add(Dense(NUM_HIDDEN_NEURONS, input_dim=3*4*monster.leg_count, activation='tanh'))
    nnet.add(Dense(NUM_HIDDEN_NEURONS, activation='tanh'))
    nnet.add(Dense(monster.get_num_actions(), activation='linear'))

    # nnet.summary()

    # nnet.compile(optimizer='rmsprop', loss="mean_squared_error", metrics=["mean_squared_error"])

    nnet.compile(optimizer='rmsprop', loss="mean_squared_error")

    # X = np.asarray([tests[0].inputs], type=np.float32)
    # Y = np.asarray([tests[0].outputs], type=np.float32)
    # X = np.asarray([[5]*(3 * monster.num_joints)], dtype=np.float32)
    # Y = np.asarray([[1]*monster.get_num_actions()], dtype=np.float32)
    # nnet.fit(X, Y, epochs=EPOCH, batch_size=min(1, CUR_TESTS_NUMBER), verbose=2)
    # predictions = nnet.predict(X)
    # print(predictions)

def SaveNNet():
    global nnet, nnets_dir_str, creature_name, nnet_finame_end

    cur_time = datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')
    nnet.save(filepath=os.path.join(nnets_dir_str, creature_name + "_" + cur_time + nnet_finame_end))

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
    global tests
    if len(tests) >= TOTAL_TESTS_NUMBER:
        del tests[0]
    tests.append(Test(inp, outp))

def InitTests():
    global init_tests_count, monster, tests
    with open("tests1.txt", "r") as fin:
        data = fin.readlines()

        actions_count = monster.get_num_actions()

        for i in range(0, len(data), 6):
            leg_id = int(data[i].strip())
            angs = list(map(float, data[i+1].strip().split()))
            inps = list(map(float, data[i+2].strip().split()))
            all_dist = float(data[i+3].strip())

            state_id = None
            for k in range(len(angs)):
                state_id = monster.legs[leg_id].find_state(k, angs[k])
                if (state_id == None):
                    print("Error!!!!")

                for j in range(leg_id):
                    state_id += len(monster.legs[j].coxa_states) - 1
                    state_id += len(monster.legs[j].femur_states) - 1
                    state_id += len(monster.legs[j].tibia_states) - 1
                if (k != 0):
                    state_id += len(monster.legs[leg_id].coxa_states) - 1
                if (k != 1):
                    state_id += len(monster.legs[leg_id].femur_states) - 1

                outps = [0.0 for i in range(actions_count)]
                # if (k == 2):
                outps[state_id] = all_dist

                AddTest(copy.deepcopy(inps), copy.deepcopy(outps))

        init_tests_count = len(tests)

def GetReward():
    global k_reward, used_reward, monster, prev_dist, same_action_count

    res = 0
    for i in range(len(used_reward)):
        rew = {
            ALL_DIST: math.fabs(monster.get_cur_delta_distance()),
            PREV_STEP_DIST: math.fabs(prev_dist - monster.get_cur_delta_distance()),
            CENTER_OF_BODY_Z: -k_reward["CENTER_OF_BODY_Z"] / max(1.0, monster.get_center_of_body()[2]),
            REPEAT_ACTION: -k_reward["REPEAT_ACTION"] / max(1.0, same_action_count)
        }[i]
        res += rew

        print(res)

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

    if random.random() < 0.3:
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
    if (cur_tick % T == 0):
        SaveNNet()

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
    ax.set_xlim3d([-10.0, 50.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-10.0, 40.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0.0, 30.0])
    ax.set_zlabel('Z')

def Draw(tick, coord_syst_lines, leg_lines, point_annotation):
    global monster #, legs , coord_system, steps, pause

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

    # time.sleep(pause[tick] / 10000000)

    return coord_syst_lines, leg_lines, point_annotation

if __name__ == "__main__":
    creature_name = "Hexapod_ad"
    crfilename = os.path.join(creature_dir_str, "Hexapod2" + creature_finame_end)
    CreatureInitializationFromFile(crfilename)
    # CreatureInitialization()

    NNet()
    # nnet = load_model('models/model_20200530-23-08-36.h5')

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




