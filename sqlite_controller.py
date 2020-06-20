import sqlite3
import json
import random
from datetime import datetime


def create_connection(db_file):
    """ Create a database connection to the SQLite database specified by db_file

    :param
    db_file: database file

    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Exception as e:
        print("DB connect error: {}".format(e))

    return conn

def create_db(connect, cursor):
    cursor.execute('''CREATE TABLE IF NOT EXISTS neuro_net
                 (nnet_id integer PRIMARY KEY AUTOINCREMENT,
                 name text,
                 config_json text)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS learning_algorithm
                (algo_id integer PRIMARY KEY AUTOINCREMENT,
                name text,
                config_json text)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS creature
                (creature_id integer PRIMARY KEY AUTOINCREMENT,
                name text,
                config_json text)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS system_config
                    (sys_conf_id integer PRIMARY KEY AUTOINCREMENT,
                    name text,
                    nnet_id integer,
                    algo_id integer,
                    creature_id integer,
                    running_type text)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS history
                    (id integer PRIMARY KEY AUTOINCREMENT,
                    sys_conf_id integer,
                    weights text,
                    biases text,
                    additional_data text,
                    ts text)''')
    connect.commit()


def add_creature(connect, cursor, data):
    cursor.executemany("INSERT INTO creature(name, config_json) VALUES (?,?)", data)
    connect.commit()

def add_neuro_net(connect, cursor, data):
    cursor.executemany("INSERT INTO neuro_net(name, config_json) VALUES (?,?)", data)
    connect.commit()

def add_learning_algorithm(connect, cursor, data):
    cursor.executemany("INSERT INTO learning_algorithm(name, config_json) VALUES (?,?)", data)
    connect.commit()

def add_system_config(connect, cursor, data):
    cursor.executemany("INSERT INTO system_config(name, nnet_id, algo_id, creature_id, running_type) VALUES (?,?,?,?,?)", data)
    connect.commit()

def add_history(connect, cursor, data):
    cursor.executemany("INSERT INTO history(sys_conf_id, weights, biases, additional_data, ts) VALUES (?,?,?,?,?)", data)
    connect.commit()

def get_creatures(cursor):
    cursor.execute("SELECT * FROM creature")
    return cursor.fetchall()

def get_creature(cursor, name, config_json):
    cursor.execute("SELECT * FROM creature WHERE name=\'" + name + "\' and config_json=\'"+ config_json +"\'")
    return cursor.fetchall()

def get_neuro_nets(cursor):
    cursor.execute("SELECT * FROM neuro_net")
    return cursor.fetchall()

def get_neuro_net(cursor, name, config_json):
    cursor.execute("SELECT * FROM neuro_net WHERE name=\'" + name + "\' and config_json=\'"+ config_json +"\'")
    return cursor.fetchall()

def get_learning_algorithms(cursor):
    cursor.execute("SELECT * FROM learning_algorithm")
    return cursor.fetchall()

def get_learning_algorithm(cursor, name, config_json):
    cursor.execute("SELECT * FROM learning_algorithm WHERE name=\'" + name + "\' and config_json=\'"+ config_json +"\'")
    return cursor.fetchall()

def get_system_configs(cursor):
    cursor.execute("SELECT * FROM system_config")
    return cursor.fetchall()

def get_system_config(cursor, name, nnet_id, algo_id, creature_id, running_type):
    cursor.execute("SELECT * FROM system_config WHERE name=\'" + name + "\'" +
                   " and nnet_id=" + str(nnet_id) +
                   " and algo_id=" + str(algo_id) +
                   " and creature_id=" + str(creature_id) +
                   " and running_type=\'" + running_type + "\'"
                   )
    return cursor.fetchall()

def get_histories(cursor):
    cursor.execute("SELECT * FROM history")
    return cursor.fetchall()

if __name__ == "__main__":
    connect = create_connection('ModelDB.db')

    with connect:
        cursor = connect.cursor()
        create_db(connect, cursor)


# def create_db(connect, cursor):
#     cursor.execute('''CREATE TABLE IF NOT EXISTS dataset
#                  (id integer PRIMARY KEY AUTOINCREMENT,
#                  leg_id integer,
#                  end_point text,
#                  angs text,
#                  ts text)''')
#
#     connect.commit()
#
# def add_dataset_row(connect, cursor, leg_id, end_point, angs):
#     ts = str(datetime.now())
#     cursor.executemany("INSERT INTO dataset(leg_id, end_point, angs, ts) VALUES (?,?,?,?)", [(leg_id, str(end_point), str(angs), ts)])
#     connect.commit()
#
# def get_dataset(connect, cursor, ts1, ts2):
#     cursor.execute("SELECT * FROM dataset where ts between ts1 and ts2")
#     return cursor.fetchall()
#
# def get_all_dataset(connect, cursor):
#     cursor.execute("SELECT * FROM dataset")
#     return cursor.fetchall()
# ----------------------------------------------------------------------------------------------------------------------

# def get_data_neuro_net():
#     name = "nnet_1"
#     config_json = json.dumps({"Inputs count": 18,
#                               "Layers count": 3,
#                               "Neurons count": "54 100 54",
#                               "Outputs count": 36,
#                               "Act Functions": "LINE TANH TANH"})
#     return [(name, str(config_json))]
#
# def add_neuro_net(connect, cursor, data):
#     cursor.executemany("INSERT INTO neuro_net(name, config_json) VALUES (?,?)", data)
#     connect.commit()
#
# def add_learning_algorithm(connect, cursor, data):
#     cursor.executemany("INSERT INTO learning_algorithm(name, config_json) VALUES (?,?)", data)
#     connect.commit()
#
# def add_creature(connect, cursor, data):
#     cursor.executemany("INSERT INTO creature(name, config_json) VALUES (?,?)", data)
#     connect.commit()
#
# def add_system_config(connect, cursor, data):
#     cursor.executemany("INSERT INTO system_config(nnet_id, algo_id, creature_id, running_type) VALUES (?,?,?,?)", data)
#     connect.commit()
#
# def add_history(connect, cursor, data):
#     cursor.executemany("INSERT INTO history(sys_conf_id, weights, biases, additional_data, ts) VALUES (?,?,?,?,?)", data)
#     connect.commit()
#
# def get_data_neuro_net():
#     name = "nnet_1"
#     config_json = json.dumps({"Inputs count": 18,
#                               "Layers count": 3,
#                               "Neurons count": "54 100 54",
#                               "Outputs count": 36,
#                               "Act Functions": "LINE TANH TANH"})
#     return [(name, str(config_json))]
#
# def get_data_learning_algorithm():
#     name = "Hexapod 1"
#     # config_json = json.dumps({"Type": "RMS",
#     #                           "Epoch": 100,
#     #                           "Train_Accuracy": 0.00001,
#     #                           "RMS Gamma": 0.95,
#     #                           "RMS Learning rate": 0.001,
#     #                           "RMS Accuracy": 0.00001,
#     #                           "QGamma": 0.9})
#     # config_json = json.dumps({"Type": "GRAD",
#     #                           "Epoch": 100,
#     #                           "Train_Accuracy": 0.00001,
#     #                           "Min grad": 0.95,
#     #                           "Learning rate": 0.001,
# 	#							"QGamma": 0.9})
#     config_json = json.dumps({"Type": "RPROP",
#                               "Epoch": 100,
#                               "Train_Accuracy": 0.00001,
#                               "Min grad": 0.95,
# 							  "QGamma": 0.9})
#     return [(name, str(config_json))]
#
# def get_data_creature():
#     name = "RMS Algo 1"
#     config_json = json.dumps({"Legs count": 6,
#                               "Servo count": 18,
#                               "Start pos":
#     {"front": ["down", "down", "down", "down", "down", "down"], "top": ["middle", "middle", "middle", "middle", "middle", "middle"],
#     "frontang": ["90", "90", "90", "90", "90", "90"], "topang": ["90", "90", "90", "90", "90", "90"], "delay": 1000},
#     })
#     return [(name, str(config_json))]
#
# def get_data_system_config():
#     nnet_id = 1
#     # algo_id = 1
#     # algo_id = 2
#     algo_id = 3
#     creature_id = 1
#     running_type = "TRAIN"  #"RUN"  # "TRAIN"
#     return [(nnet_id, algo_id, creature_id, running_type)]
#
# def get_data_history():
#     movement_algorithm = [
#         # {"front": ["down", "down", "down", "down", "down", "down"], "top": ["middle", "middle", "middle", "middle", "middle", "middle"],
#         # "frontang": ["90", "90", "90", "90", "90", "90"], "topang": ["90", "90", "90", "90", "90", "90"], "delay": 1000}, // init
#         {"front": ["up", "down", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["middle", "middle", "middle", "middle", "middle", "middle"],
#          "frontang": ["110", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["90", "90", "90", "90", "90", "90"],
#          "delay": 250},
#         {"front": ["up", "down", "down", "down", "down", "down", "", "", "", "", "", "", "", "", "", "", "", ""],
#          "top": ["forward", "middle", "middle", "middle", "middle", "middle"],
#          "frontang": ["110", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["105", "90", "90", "90", "90", "90"],
#          "delay": 250},
#         {"front": ["down", "down", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["forward", "middle", "middle", "middle", "middle", "middle"],
#          "frontang": ["90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["105", "90", "90", "90", "90", "90"],
#          "delay": 500},
#
#         {"front": ["down", "down", "up", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["forward", "middle", "middle", "middle", "middle", "middle"],
#          "frontang": ["90", "90", "110", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["105", "90", "90", "90", "90", "90"],
#          "delay": 250},
#         {"front": ["down", "down", "up", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["forward", "middle", "forward", "middle", "middle", "middle"],
#          "frontang": ["90", "90", "110", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["105", "90", "105", "90", "90", "90"],
#          "delay": 250},
#         {"front": ["down", "down", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["forward", "middle", "forward", "middle", "middle", "middle"],
#          "frontang": ["90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["105", "90", "105", "90", "90", "90"],
#          "delay": 500},
#
#         {"front": ["down", "down", "down", "down", "up", "down", "", "", "", "", "", ""],
#          "top": ["forward", "middle", "forward", "middle", "middle", "middle"],
#          "frontang": ["90", "90", "90", "90", "70", "90", "90", "90", "90", "90", "90", "90"], "topang": ["105", "90", "105", "90", "90", "90"],
#          "delay": 250},
#         {"front": ["down", "down", "down", "down", "up", "down", "", "", "", "", "", ""],
#          "top": ["forward", "middle", "forward", "middle", "forward", "middle"],
#          "frontang": ["90", "90", "90", "90", "70", "90", "90", "90", "90", "90", "90", "90"], "topang": ["105", "90", "105", "90", "75", "90"],
#          "delay": 250},
#         {"front": ["down", "down", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["forward", "middle", "forward", "middle", "forward", "middle"],
#          "frontang": ["90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["105", "90", "105", "90", "75", "90"],
#          "delay": 500},
#         # ---
#         {"front": ["down", "down", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["middle", "middle", "forward", "middle", "forward", "middle"],
#          "frontang": ["90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["90", "90", "105", "90", "75", "90"],
#          "delay": 50},
#         {"front": ["down", "down", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["middle", "middle", "middle", "middle", "forward", "middle"],
#          "frontang": ["90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["90", "90", "90", "90", "75", "90"], "delay": 50},
#         {"front": ["down", "down", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["middle", "middle", "middle", "middle", "middle", "middle"],
#          "frontang": ["90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["90", "90", "90", "90", "90", "90"], "delay": 50},
#
#         {"front": ["down", "down", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["middle", "back", "middle", "middle", "middle", "middle"],
#          "frontang": ["90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["90", "75", "90", "90", "90", "90"], "delay": 50},
#         {"front": ["down", "down", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["middle", "back", "middle", "back", "middle", "middle"],
#          "frontang": ["90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["90", "75", "90", "105", "90", "90"],
#          "delay": 50},
#         {"front": ["down", "down", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["middle", "back", "middle", "back", "middle", "back"],
#          "frontang": ["90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["90", "75", "90", "105", "90", "105"],
#          "delay": 1000},
#         # ---
#
#         {"front": ["down", "up", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["middle", "back", "middle", "back", "middle", "back"],
#          "frontang": ["90", "110", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["90", "75", "90", "105", "90", "105"],
#          "delay": 250},
#         {"front": ["down", "up", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["middle", "forward", "middle", "back", "middle", "back"],
#          "frontang": ["90", "110", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["90", "105", "90", "105", "90", "105"],
#          "delay": 250},
#         {"front": ["down", "down", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["middle", "forward", "middle", "back", "middle", "back"],
#          "frontang": ["90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["90", "105", "90", "105", "90", "105"],
#          "delay": 500},
#
#         {"front": ["down", "down", "down", "up", "down", "down", "", "", "", "", "", ""],
#          "top": ["middle", "forward", "middle", "back", "middle", "back"],
#          "frontang": ["90", "90", "90", "70", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["90", "105", "90", "105", "90", "105"],
#          "delay": 250},
#         {"front": ["down", "down", "down", "up", "down", "down", "", "", "", "", "", ""],
#          "top": ["middle", "forward", "middle", "forward", "middle", "back"],
#          "frontang": ["90", "90", "90", "70", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["90", "105", "90", "75", "90", "105"],
#          "delay": 250},
#         {"front": ["down", "down", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["middle", "forward", "middle", "forward", "middle", "back"],
#          "frontang": ["90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["90", "105", "90", "75", "90", "105"],
#          "delay": 500},
#
#         {"front": ["down", "down", "down", "down", "down", "up", "", "", "", "", "", ""],
#          "top": ["middle", "forward", "middle", "forward", "middle", "back"],
#          "frontang": ["90", "90", "90", "90", "90", "70", "90", "90", "90", "90", "90", "90"], "topang": ["90", "105", "90", "75", "90", "105"],
#          "delay": 250},
#         {"front": ["down", "down", "down", "down", "down", "up", "", "", "", "", "", ""],
#          "top": ["middle", "forward", "middle", "forward", "middle", "forward"],
#          "frontang": ["90", "90", "90", "90", "90", "70", "90", "90", "90", "90", "90", "90"], "topang": ["90", "105", "90", "75", "90", "75"],
#          "delay": 250},
#         {"front": ["down", "down", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["middle", "forward", "middle", "forward", "middle", "forward"],
#          "frontang": ["90", "90", "90", "90", "90", "70", "90", "90", "90", "90", "90", "90"], "topang": ["90", "105", "90", "75", "90", "75"],
#          "delay": 500},
#         # ---
#         {"front": ["down", "down", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["middle", "middle", "middle", "forward", "middle", "forward"],
#          "frontang": ["90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["90", "90", "90", "75", "90", "75"], "delay": 50},
#         {"front": ["down", "down", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["middle", "middle", "middle", "middle", "middle", "forward"],
#          "frontang": ["90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["90", "90", "90", "90", "90", "75"], "delay": 50},
#         {"front": ["down", "down", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["middle", "middle", "middle", "middle", "middle", "middle"],
#          "frontang": ["90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["90", "90", "90", "90", "90", "90"], "delay": 50},
#
#         {"front": ["down", "down", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["back", "middle", "middle", "middle", "middle", "middle"],
#          "frontang": ["90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["75", "90", "90", "90", "90", "90"], "delay": 50},
#         {"front": ["down", "down", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["back", "middle", "back", "middle", "middle", "middle"],
#          "frontang": ["90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["75", "90", "90", "75", "90", "90"], "delay": 50},
#         {"front": ["down", "down", "down", "down", "down", "down", "", "", "", "", "", ""],
#          "top": ["back", "middle", "back", "middle", "middle", "middle"],
#          "frontang": ["90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"], "topang": ["75", "90", "75", "90", "90", "90"],
#          "delay": 1000}
#     ]
#
#     res = []
#
#     # sys_conf_id = 1
#     # sys_conf_id = 2
#     sys_conf_id = 3
#     weights = []
#     biases = []
#
#     pos = 0
#     cur_dist = 0
#     max_dist = 0
#     for i in range(1000000):
#         prev_step_dist = random.uniform(1, 10)
#         p = random.random()
#         if(p < 0.2):
#             prev_step_dist = -prev_step_dist
#         cur_dist += prev_step_dist
#         max_dist = max(cur_dist, max_dist)
#         cgz = prev_step_dist = -1.0/random.uniform(10, 15)
#         fall = -1.0/random.uniform(5, 10)
#         cur_reward = cur_dist + prev_step_dist + cgz + fall
#
#
#         additional_data = json.dumps({"Step": movement_algorithm[pos],
#                                         "Cur dist": cur_dist,
#                                         "Max dist": max_dist,
#                                         "Cur reward":cur_reward,
#                                         "ALL_DIST": cur_dist,
#                                         "PREV_STEP_DIST": prev_step_dist,
#                                         "-k/CENTER_OF_GRAVITY_Z": cgz,
#                                         "-k/FALLING": fall})
#         ts = str(datetime.now())
#
#         res.append((sys_conf_id, str(weights), str(biases), str(additional_data), ts))
#
#         pos+=1
#         if (pos == len(movement_algorithm)):
#             pos = 0
#
#     return res




# if __name__ == "__main__":
#     connect = create_connection('3DModelDB.db')
#
#     with connect:
#         cursor = connect.cursor()
#         # create_db(connect, cursor)
#         # cursor.execute("DROP TABLE dataset")
#         # connect.commit()
#
#         dts = get_all_dataset(connect, cursor)
#         [print(str(el) + '\n') for el in dts]










        # # add_neuro_net(connect, cursor, get_data_neuro_net())
        # # add_learning_algorithm(connect, cursor, get_data_learning_algorithm())
        # # add_creature(connect, cursor, get_data_creature())
        # # add_system_config(connect, cursor, get_data_system_config())
        # add_history(connect, cursor, get_data_history())
        #
        # # cursor.execute("SELECT * FROM neuro_net limit 1")
        # # row = cursor.fetchall()
        # # print(row)
        # # cursor.execute("SELECT * FROM learning_algorithm limit 1")
        # # row = cursor.fetchall()
        # # print(row)
        # # cursor.execute("SELECT * FROM creature limit 1")
        # # row = cursor.fetchall()
        # # print(row)
        # # cursor.execute("SELECT * FROM system_config limit 1")
        # # row = cursor.fetchall()
        # # print(row)
        # cursor.execute("SELECT ts FROM history limit 1000")
        # row = cursor.fetchall()
        # print(row)
