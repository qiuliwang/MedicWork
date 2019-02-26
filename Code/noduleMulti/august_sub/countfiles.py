import os
import numpy as np 

import csvTools

# labels = csvTools.readCSV('files/malignancy.csv')

# lobcount1 = []
# lobcount2 = []
# lobcount3 = []
# lobcount4 = []
# lobcount5 = []

# for onenodule in labels:
#     sign = round(float(onenodule[21]))
#     if sign == 1:
#         lobcount1.append(onenodule)
#     elif sign == 2:
#         lobcount2.append(onenodule)
#     elif sign == 3:
#         lobcount3.append(onenodule)
#     elif sign == 4:
#         lobcount4.append(onenodule)
#     elif sign == 5:
#         lobcount5.append(onenodule)

# print(len(lobcount1), len(lobcount2), len(lobcount3), len(lobcount4), len(lobcount5))

# malcount1 = 0
# malcount2 = 0
# malcount3 = 0
# malcount4 = 0
# malcount5 = 0

# for one in lobcount1:
#     sign = round(float(one[29]))
#     if sign == 1:
#         malcount1 += 1
#     elif sign == 2:
#         malcount2 += 1
#     elif sign == 3:
#         malcount3 += 1
#     elif sign == 4:
#         malcount4 += 1
#     elif sign == 5:
#         malcount5 += 1

# print(malcount1, malcount2, malcount3, malcount4, malcount5)

# malcount1 = 0
# malcount2 = 0
# malcount3 = 0
# malcount4 = 0
# malcount5 = 0

# for one in lobcount2:
#     sign = round(float(one[29]))
#     if sign == 1:
#         malcount1 += 1
#     elif sign == 2:
#         malcount2 += 1
#     elif sign == 3:
#         malcount3 += 1
#     elif sign == 4:
#         malcount4 += 1
#     elif sign == 5:
#         malcount5 += 1

# print(malcount1, malcount2, malcount3, malcount4, malcount5)

# malcount1 = 0
# malcount2 = 0
# malcount3 = 0
# malcount4 = 0
# malcount5 = 0

# for one in lobcount3:
#     sign = round(float(one[29]))
#     if sign == 1:
#         malcount1 += 1
#     elif sign == 2:
#         malcount2 += 1
#     elif sign == 3:
#         malcount3 += 1
#     elif sign == 4:
#         malcount4 += 1
#     elif sign == 5:
#         malcount5 += 1

# print(malcount1, malcount2, malcount3, malcount4, malcount5)

# malcount1 = 0
# malcount2 = 0
# malcount3 = 0
# malcount4 = 0
# malcount5 = 0

# for one in lobcount4:
#     sign = round(float(one[29]))
#     if sign == 1:
#         malcount1 += 1
#     elif sign == 2:
#         malcount2 += 1
#     elif sign == 3:
#         malcount3 += 1
#     elif sign == 4:
#         malcount4 += 1
#     elif sign == 5:
#         malcount5 += 1

# print(malcount1, malcount2, malcount3, malcount4, malcount5)

# malcount1 = 0
# malcount2 = 0
# malcount3 = 0
# malcount4 = 0
# malcount5 = 0

# for one in lobcount5:
#     sign = round(float(one[29]))
#     if sign == 1:
#         malcount1 += 1
#     elif sign == 2:
#         malcount2 += 1
#     elif sign == 3:
#         malcount3 += 1
#     elif sign == 4:
#         malcount4 += 1
#     elif sign == 5:
#         malcount5 += 1

# print(malcount1, malcount2, malcount3, malcount4, malcount5)

datadir = '1245/'

filelist = os.listdir(datadir)
print(len(filelist))

counthigh = 0
countlow = 0
for onefile in filelist:
    if 'high' in onefile:
        counthigh += 1
    elif 'low' in onefile:
        countlow += 1
print(counthigh)
print(countlow)


'''
lob
1430
(225, 431, 677, 93, 4)

783
(40, 131, 423, 170, 19)

292
(11, 20, 102, 99, 60)

92
(3, 9, 29, 25, 26)

35
(4, 9, 14, 3, 5)



spi
1644
(225, 431, 677, 93, 4)

648
(40, 131, 423, 170, 19)

194
(11, 20, 102, 99, 60)

101
(3, 9, 29, 25, 26)

45
(4, 9, 14, 3, 5)

'''
