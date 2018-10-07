'''
Given information of 18 polluntant(factor) in the air over 12*20 days,
24 hours per day, how to predict the PM2.5 value based on the values of 18
factors in the previous 9 hours.
'''



import csv
import data as data
import numpy as np
from numpy.linalg import inv
import random
import math
import sys


#   READ DATA
data = []
# every slot in the list 'data' stores the information for one pollutant
for i in range(18):
    data.append([])

n_row = 0
text = open('/Users/****/Desktop/ML_NTU_2017Fall/HW1-Predicting PM2.5/train.csv', 'r', encoding='big5')
row = csv.reader(text, delimiter=",")
print(row)
for r in row:
    # no pollutant information in row 0
    if n_row != 0:
        # pollutant information in column 3 to 27
        for i in range(3, 27):
            if r[i] != "NR":
                data[(n_row - 1) % 18].append(float(r[i]))
            else:
                data[(n_row - 1) % 18].append(float(0))
    n_row = n_row + 1
text.close()


#   PARESE DATA INTO X AND Y
x = []
y = []
# for i in 12 months
for i in range(12):
    # for 471 sets of data in successive 10 hours(inter-day)
    for j in range(471):
        x.append([])
        # for 18 kinds of pollutant
        for t in range(18):
            # parse previous 9 hours into x
            for s in range(9):
                # for each month, there are 480 data each pollutant
                x[471 * i + j].append(data[t][480 * i + j + s])
        y.append(data[9][480 * i + j + 9])
x = np.array(x)
y = np.array(y)

# add square term
# x = np.concatenate((x, x**2), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
print(x)


#   DEFINE WEIGHT AND OTHER PARAMETER
w = np.zeros(len(x[0]))
l_rate = 10
repeat = 10000

# w = np.matmul(np.matmul(inv(np.matmul(x.transpose(), x)), x.transpose()), y)
# print(w)

#   START TRAINING
x_t = x.transpose()
s_gra = np.zeros(len(x[0]))
print(x_t)

for i in range(repeat):
    hypo = np.dot(x, w)
    loss = hypo - y
    cost = np.sum(loss ** 2) / len(x)
    cost_a = math.sqrt(cost)
    gra = np.dot(x_t, loss)
    s_gra += gra ** 2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra / ada
    print('iteration: %d | Cost: %f  ' % (i, cost_a))


#   Save model
np.save('/Users/****/Desktop/ML_NTU_2017Fall/HW1-Predicting PM2.5/model.npy', w)
#   Read model
w = np.load('/Users/****/Desktop/ML_NTU_2017Fall/HW1-Predicting PM2.5/model.npy')


#   READ TEST DATA
test_x = []
n_row = 0
text = open('/Users/****/Desktop/ML_NTU_2017Fall/HW1-Predicting PM2.5/test.csv', 'r')
row = csv.reader(text, delimiter=",")

for r in row:
    if n_row % 18 == 0:
        test_x.append([])
        for i in range(2, 11):
            test_x[n_row // 18].append(float(r[i]))
    else:
        for i in range(2, 11):
            if r[i] != "NR":
                test_x[n_row // 18].append(float(r[i]))
            else:
                test_x[n_row // 18].append(0)
    n_row = n_row + 1
text.close()
test_x = np.array(test_x)
print(test_x)
# add square term
# test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis=1)
print(test_x)


#  PREDIT BASED ON MODEL
ans = []
for i in range(len(test_x)):
    ans.append(["id_" + str(i)])
    a = np.dot(w, test_x[i])
    ans[i].append(a)

filename = "/Users/****/Desktop/ML_NTU_2017Fall/HW1-Predicting PM2.5/sampleSubmission.csv"
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "value"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()
