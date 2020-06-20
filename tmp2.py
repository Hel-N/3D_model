# fin = open("Res_angs/Hexapod_prd_cob_ra.txt", "r")
# data = fin.readlines()
#
# for i in range(len(data)):
#     d = data[i].strip().split()
#     print("{:.5f}".format(float(d[-1])))

import h5py
f = h5py.File('models_points/Hexapod_prd_20200610-06-30-46.h5', "r+")
data_p = f.attrs['training_config']
data_p = data_p.decode().replace("learning_rate","lr").encode()
f.attrs['training_config'] = data_p
f.close()
