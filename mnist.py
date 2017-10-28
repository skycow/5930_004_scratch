from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime

mnist = input_data.read_data_sets("/tmp/data/")

X_train = mnist.train.images
Y_train = mnist.train.labels.astype("int")

X_test = mnist.test.images
Y_test = mnist.test.labels.astype("int")

batch_size = 100
num_mats = 28*28
num_hid_neur = 300
num_outputs = 10
step_size = 0.1
iterations = 500

X_train_small = X_train[0:batch_size]
Y_train_small = Y_train[0:batch_size]

Y_train_small_mat = np.zeros((batch_size,num_outputs))

for r in range(batch_size):
    Y_train_small_mat[r][Y_train_small[r]] = 1

W = [];

W.append(np.random.rand(num_mats, num_hid_neur)*0.5)
W.append(np.random.rand(num_hid_neur, num_outputs)*0.5)

def myrelucomp(val):
    if(val < 0):
        return 0
    else:
        return val

def relu(matin):
    mat = np.zeros(matin.shape)
    rows, cols = mat.shape
    for r in range(rows):
        for c in range(cols):
            mat[r][c] = myrelucomp(matin[r][c])
    return mat

def softmax(matin):
    mat = np.zeros(matin.shape)
    rows, cols = mat.shape
    for r in range(rows):
        for c in range(cols):
            alpha = np.ndarray.max(matin[r])
            mat[r][c] = np.exp(matin[r][c]-alpha)
    for r in range(rows):
        softmaxsum = 0
        for cc in mat[r]:
            softmaxsum = softmaxsum + cc
        for c in range(cols):
            if softmaxsum != 0:
                mat[r][c] = mat[r][c]/softmaxsum
            else:
                mat[r][c] = 0
    return mat

def myreluderivcomp(val):
    if(val <= 0):
        return 0
    else:
        return 1

def reluderiv(matin):
    mat = np.zeros(matin.shape)
    rows, cols = mat.shape
    for r in range(rows):
        for c in range(cols):
            mat[r][c] = myreluderivcomp(matin[r][c])
    return mat

def jerror(jyhat, jy):
    return (1/10)*np.sum(jyhat-jy)**2

def fwdprop(mat, W):
    z1 = np.dot(mat, W[0])
    a1 = relu(z1)
    z2 = np.dot(a1, W[1])
    yhat = softmax(z2)
    return yhat

def backprop(mat, backy, W, iteration, rmserrors):

    #fwd prop
    z1 = np.dot(mat, W[0])
    a1 = relu(z1)
    z2 = np.dot(a1, W[1])
    yhat = softmax(z2)

    #bck prop
    delta3 = yhat-backy
    dJdw2 = np.dot(a1.T, delta3)

    delta2 = np.dot(delta3, W[1].T)*reluderiv(z1)
    dJdw1 = np.dot(mat.T, delta2)
    return dJdw1, dJdw2,

rmserrors = []
yout = []

for yiter in Y_test:
    yout.append(yiter)
    
yout_mat = np.zeros((Y_test.shape[0],num_outputs))
for r in range(batch_size):
    yout_mat[r][Y_test[r]] = 1    

for i in range(iterations):
    #pick a batch
    samp_index = random.sample(range(0, X_train.shape[0]-1), batch_size)

    X_train_batch = np.zeros((batch_size,28*28))

    for index in range(0,len(samp_index)-1):
        X_train_batch[index] = X_train[samp_index[index]]

    Y_train_batch_mat = np.zeros((batch_size,num_outputs))
    for r in range(batch_size):
        Y_train_batch_mat[r][Y_train[samp_index[r]]] = 1

    #do back prop
    dJdw1, dJdw2  = backprop(X_train_batch, Y_train_batch_mat, W, i, rmserrors)

    #update weights
    W[0] = W[0] - step_size * dJdw1/np.max(np.fabs(dJdw1))
    W[1] = W[1] - step_size * dJdw2/np.max(np.fabs(dJdw2))

    if(i%50 == 0):
        errorcalc = jerror(fwdprop(X_test,W),yout_mat)
        print("MSE on iterration " +str(i)+": " + str(errorcalc))
        rmserrors.append(errorcalc)




res = fwdprop(X_test,W)
xout = []
for x in range(res.shape[0]):
    xout.append(np.argmax(res[x]))

comp = np.matrix(yout)-np.matrix(xout)
comp2 = comp==0

print(np.around(np.sum(comp2)/comp2.shape[1]*100,2))
plt.plot(range(0, iterations, 50),rmserrors)
