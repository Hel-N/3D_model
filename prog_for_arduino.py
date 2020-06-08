import copy
import math
import random
import sys
import time
import datetime
import serial

from creature_end_points import Creature
from common import Test

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
import numpy as np

# Arduino
# ser=serial.Serial("/dev/ttyACM0",9600)  #change ACM number as found from ls /dev/tty/ACM*
# ser.baudrate=9600

DBL_MAX = 1.79769e+308

# fout_res = open("tmp_res.txt", "w")

# logout("log.txt")
# testout("test.txt")

# res_dir_str = "Res\\"
# wb_finame_end = "_weights_and_biases.txt"
# curcr_finame_end = "_cur_creature.txt"
# dist_finame_end = "_dist.txt"
# run_dist_finame_end = "_run_dist.txt"

# nnets_dir_str = "NNets\\"
# creatures_dir_str = "Creatures\\"
# nnet_finame_end = "_nnet.txt"
# creature_finame_end = "_creature.txt"

# NNet Config------------------------------------------------------
nnet_name = ""
creature_name = ""
NUM_HIDDEN_LAYERS = 2
NUM_HIDDEN_NEURONS = 100
# ACT_FUNC = TANH
TOTAL_TESTS_NUMBER = 1000
CUR_TESTS_NUMBER = 100
EPOCH = 10
# TRAINING_TYPE = RMS

# For RMS
RMS_GAMMA = 0.95
RMS_LEARN_RATE = 0.001
RMS_EPS = 1e-8

QGAMMA = 0.9 # Коэффициент доверия
TICK_COUNT = 10000
TRAIN_EPS = 0.001

LearningRate = 0.01 # Для алгоритма обратного распространения

ALL_DIST = 0
PREV_STEP_DIST = 1
CENTER_OF_GRAVITY_Z = 2
FALLING = 3
HEAD_Y = 4
used_reward = []
k_reward = {"CENTER_OF_GRAVITY_Z":0, "FALLING":0, "HEAD_Y":0}

recovery_from_falling = False

T = 100 #Период печати весов

# Для модели------------------------------------------------
# start_joints = list(map(Point, []))
# start_states = []

reward = 0.0
prev_dist = 0.0
cur_tick = 0

Q = []
prevQ = []
prev_action = -1
first_step = True

inputs = []
prev_inputs = []

monster = None

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
    with open("tests.txt", "r") as fin:
        data = fin.readlines()

        actions_count = monster.get_num_actions()

        for i in range(0, len(data), 6):
            leg_id = int(data[i].strip())
            end_point = list(map(float, data[i+1].strip().split()))
            inps = list(map(float, data[i+2].strip().split()))
            all_dist = float(data[i+3].strip())

            state_id = monster.legs[leg_id].find_state(end_point)
            if (state_id == None):
                print("Error!!!!")

            for j in range(leg_id):
                state_id += len(monster.legs[j].states) - 1

            outps = [0.0 for i in range(actions_count)]
            outps[state_id] = all_dist

            AddTest(copy.deepcopy(inps), copy.deepcopy(outps))

        init_tests_count = len(tests)

def GetReward():
    global k_reward
    global used_reward
    global monster
    global prev_dist
    # res = 0
    # for i in range(len(used_reward)):
        # rew = {
        #     ALL_DIST: math.fabs(monster.cur_delta_distance()),
        #     PREV_STEP_DIST: math.fabs(prev_dist - monster.cur_delta_distance()),
        #     CENTER_OF_GRAVITY_Z: -k_reward["CENTER_OF_GRAVITY_Z"] / max(1.0, monster.center_of_gravity_y()),
        #     FALLING: -k_reward["FALLING"] / max(1.0, monster.falling),
        #     HEAD_Y: -k_reward["HEAD_Y"] / max(1.0, monster.head_y)
        # }[i]
        # res += rew

    # res = math.fabs(monster.get_cur_delta_distance())
    res = math.fabs(prev_dist - monster.get_cur_delta_distance())
    # res = (prev_dist - monster.get_cur_delta_distance())
    return res

def SendAngsToArduino(action):
    global monster, ser

    leg_id, _ = monster.get_action(action)
    angs = monster.legs[leg_id].servo_angs()

    for i in range(len(angs)):
        ser.write(str.encode("{0} {1} {2}#".format(leg_id, i, angs[i])))
        time.sleep(0.5)
    time.sleep(0.5)

def DoNextStep():
    global recovery_from_falling, monster, prev_dist, \
        prev_inputs, inputs, reward, cur_tick, first_step, \
        Q, prevQ, prev_action, tests, res_dir_str, nnet_name, \
        dirname, wb_finame_end, curcr_finame_end, res_dir_str

    # if (recovery_from_falling and (not first_step)):
    #     if (monster.head_y <= monster.center_of_gravity_y()):
    #         monster.joints = start_joints
    #         monster.states = start_states
    #         SetCurDist(prev_dist)


    prev_inputs = copy.deepcopy(inputs)
    inputs = SetInputs()
    action = -1
    reward = 0.0
    reward = GetReward()
    prev_dist = monster.get_cur_delta_distance()
    cur_tick += 1

    # fout_res.write("{}     {}\n".format(cur_tick-1, prev_dist))

    print(cur_tick)
    print("All dist: {0}".format(prev_dist))

    if (not first_step):
        [Q] = nnet.predict(np.asarray([SetInputs()], dtype=np.float32))

        tmpQ = -DBL_MAX
        for i in range(len(Q)):
            if (tmpQ < Q[i]):  # and monster.can_do_action(i)):
                tmpQ = Q[i]
                action = i

        Q[prev_action] = reward + QGAMMA*tmpQ

        AddTest(prev_inputs, Q)
        # epoch = EPOCH

        # cur_tests_in = np.asarray([copy.deepcopy(tests[i].inputs) for i in range(len(tests))], dtype=np.float32)
        # cur_tests_out = np.asarray([copy.deepcopy(tests[i].outputs) for i in range(len(tests))], dtype=np.float32)
        # nnet.fit(cur_tests_in, cur_tests_out, epochs=EPOCH, batch_size=min(1, CUR_TESTS_NUMBER), verbose=0)
    else:
        [Q] = nnet.predict(np.asarray([SetInputs()], dtype=np.float32))

        tmpQ = -DBL_MAX
        for i in range(len(Q)):
            if (tmpQ < Q[i]): # and monster.can_do_action(i)):
                tmpQ = Q[i]
                action = i
        first_step = False

    if (action != prev_action): # ?
        monster.update_pos(action_num=action)
        # SendAngsToArduino(action)

    # if random.random() < 0.1:
    #     counter = 100
    #     flag_do = False
    #     for i in range(counter):
    #         action = random.randint(0, monster.get_num_actions() - 1)
    #         # if monster.can_do_action(action):
    #         flag_do = True
    #         break
    #     if flag_do:
    #         if (action != prev_action): # ?
    #             monster.update_pos(action_num=action)

    prev_action = action
    prevQ = copy.deepcopy(Q)

    #Сохранение модели
    # if (cur_tick % T == 0):
    #     nnet.save(filepath="C:/Users/Елена/Desktop/Диплом_Маг/3D_model/models/model_" + datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S') + '.h5')

    # Вывод текущей информации
    # if (cur_tick % T == 0):
    #     dirname = res_dir_str + nnet_name
    #     with open(dirname + "\\" + nnet_name + wb_finame_end, "a") as wbfout:
    #         nnet.PrintWeightsAndBiases(wbfout, false);
            # pass
        #
        # with open(dirname + "\\" + nnet_name + curcr_finame_end, "a") as crfout:
        #     monster.print_creature_joints(crfout)

def Timer(tick, coord_syst_lines, leg_lines, point_annotation):
    DoNextStep()


legs = []
coord_syst_lines = []
leg_lines = []
point_annotation = []

if __name__ == "__main__":
    # Инициализация существа
    creature_name = "hexapod"
    # crfilename = creatures_dir_str + creature_name + creature_finame_end
    # CreatureInitializationFromFile(crfilename)
    CreatureInitialization()

    nnet_name = "hexapod"
    # nnfilename = nnets_dir_str + nnet_name + nnet_finame_end
    # NNet()
    nnet = load_model('models/model_20200530-23-08-36.h5')

    print("ok")
    # InitTests()

    # msg = ser.readline()
    # print(msg)
    # while True:
    #     msg = ser.readline()
    #     print(msg)
    #     if (msg == str.encode("Ready!\r\n")):
    #         print("Arduino is ready!\n")
    #         ser.write(str.encode("Go"))
    #         msg = ser.readline()
    #         print(msg)
    #         break

    # while True:
    for i in range(10):
        DoNextStep()
        # ser.write(str.encode(msgs_array[i][j] + '#'))
        #
        # for i in range(len(msgs_array)):
        #     for j in range(len(msgs_array[i])):
        #
        #         if (msgs_array[i][j][:8] == "startpos" or msgs_array[i][j][:11] == "rotate back"):
        #             time.sleep(0.1)
        #         else:
        #             time.sleep(0.5)
        #         msg = ser.readline()
        #         print(msg)
        #     time.sleep(1)






