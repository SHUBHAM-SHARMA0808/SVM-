
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from svmutil import *

#part 1 : Linear svm
y_train, x_train = svm_read_problem('ncrna_train.csv')

reg0 = []
acc0 = []

for i in range(1, 10) :
    m = svm_train(y_train[:600], x_train[:600], '-s 0 -c {} -t 0'.format(i))
    p_label_train, p_acc_train, p_val_train = svm_predict(y_train[600:], x_train[600:], m)
    reg0.append(i)
    acc0.append(p_acc_train[0])  
    
plt.plot(reg0, acc0, 'r--')  
plt.show()  

y_test, x_test = svm_read_problem('ncrna_test.csv')
p_label_test, p_acc_test, p_val_test = svm_predict(y_test, x_test, m)

print(p_label_test)

#part 2 : RBF svm
reg = []
sigma = []
acc = []

x_train_fold1 = x_train[1000 : 1200]
y_train_fold1 = y_train[1000 : 1200]

x_train_fold2 = x_train[1200 : 1400]
y_train_fold2 = y_train[1200 : 1400]

x_train_fold3 = x_train[1400 : 1600]
y_train_fold3 = y_train[1400 : 1600]

x_train_fold4 = x_train[1600 : 1800]
y_train_fold4 = y_train[1600 : 1800]

x_train_fold5 = x_train[1800 : 2000]
y_train_fold5 = y_train[1800 : 2000]

count = 0
p = 11
for i in range(1, 11) :
    for j in range(1, 11) :
        #m0 = svm_train(y_train[:1000], x_train[:1000], '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        #m1 = svm_train(y_train_fold1, x_train_fold1, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        m1 = svm_train(y_train_fold2, x_train_fold2, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        m1 = svm_train(y_train_fold3, x_train_fold3, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        m1 = svm_train(y_train_fold4, x_train_fold4, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        m1 = svm_train(y_train_fold5, x_train_fold5, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        p2_label_fold1, p2_acc_fold1, p2_val_fold1 = svm_predict(y_train_fold1, x_train_fold1, m1)
     
        m2 = svm_train(y_train_fold1, x_train_fold1, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        #m2 = svm_train(y_train_fold2, x_train_fold2, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        m2 = svm_train(y_train_fold3, x_train_fold3, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        m2 = svm_train(y_train_fold4, x_train_fold4, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        m2 = svm_train(y_train_fold5, x_train_fold5, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        p2_label_fold2, p2_acc_fold2, p2_val_fold2 = svm_predict(y_train_fold2, x_train_fold2, m2)
     
        m3 = svm_train(y_train_fold1, x_train_fold1, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        m3 = svm_train(y_train_fold2, x_train_fold2, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        #m3 = svm_train(y_train_fold3, x_train_fold3, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        m3 = svm_train(y_train_fold4, x_train_fold4, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        m3 = svm_train(y_train_fold5, x_train_fold5, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        p2_label_fold3, p2_acc_fold3, p2_val_fold3 = svm_predict(y_train_fold3, x_train_fold3, m3)

        m4 = svm_train(y_train_fold1, x_train_fold1, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        m4 = svm_train(y_train_fold2, x_train_fold2, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        m4 = svm_train(y_train_fold3, x_train_fold3, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        #m4 = svm_train(y_train_fold4, x_train_fold4, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        m4 = svm_train(y_train_fold5, x_train_fold5, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        p2_label_fold4, p2_acc_fold4, p2_val_fold4 = svm_predict(y_train_fold4, x_train_fold4, m4)
       
        m5 = svm_train(y_train_fold1, x_train_fold1, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        m5 = svm_train(y_train_fold2, x_train_fold2, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        m5 = svm_train(y_train_fold3, x_train_fold3, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        m5 = svm_train(y_train_fold4, x_train_fold4, '-s 0 -c {} -g {} -t 2'.format(i, p-j))
        #m5 = svm_train(y_train_fold5, x_train_fold5, '-s 0 -g {} -c {} -t 2'.format(i, p-j))
        p2_label_fold5, p2_acc_fold5, p2_val_fold5 = svm_predict(y_train_fold5, x_train_fold5, m5)
        
        acc.append((p2_acc_fold1[0] + p2_acc_fold2[0] + p2_acc_fold3[0] + p2_acc_fold4[0] + p2_acc_fold5[0])/5)
        reg.append(i)
        sigma.append(p-j)
        count += 1
count 

rel_mat = np.zeros((11,11))
rel_mat    

k = 0
for i in range(1,11) :
    for j in range(1,11) :
        rel_mat[i][j] = acc[k]
        k = k+1
for i in range(1,11) :
    rel_mat[0][i] = sigma[i-1]
    rel_mat[i][0] = reg[10*(i-1)+1]

rel_mat 

max_accuracy = 0 
best_sigma = 0
best_regulaizer = 0
for i in range(1,11) :
    for j in range(1,11) :
        if rel_mat[i][j]>max_accuracy :
            max_accuracy = rel_mat[i][j]
            best_regulaizer = rel_mat[i][0]
            best_sigma = rel_mat[0][j]
print(max_accuracy) 
print(best_sigma)
print(best_regulaizer)    

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sigma, reg, acc, c='r', marker='o')
ax.set_xlabel('sigma')
ax.set_ylabel('regulizer')
ax.set_zlabel('accuracy1')
plt.show()

m6 = svm_train(y_train, x_train, '-s 0 -g {} -c {} -t 2'.format(best_sigma, best_regulaizer))

p2_label_test, p2_acc_test, p2_val_test = svm_predict(y_test, x_test, m6)

print(p2_label_test)

import csv

k = 0
file_list = []  
f = open('ncrna_test.csv', 'r')
reader = csv.reader(f)

for row in reader:
    file_list.append(row)

for i in file_list:
    l = list(i[0])
    l[0] = p2_label_test[k]
    k = k+1
    i[0] = ' '.join([str(elem) for elem in l])

filename = "newncrna_test.csv"
  
with open(filename, 'w') as csvfile: 

    csvwriter = csv.writer(csvfile) 
 
    csvwriter.writerows(file_list) 






