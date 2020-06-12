fin = open("Res_angs/Hexapod_prd_cob_ra.txt", "r")
data = fin.readlines()

for i in range(len(data)):
    d = data[i].strip().split()
    print("{:.5f}".format(float(d[-1])))
