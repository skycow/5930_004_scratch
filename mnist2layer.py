#Skyler Cowley
#ECE5930-004
#Assignment 3


from tensorflow.examples.tutorials.mnist import input_data
#import matplotlib.pyplot as plt
#import matplotlib as mpl
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
num_hid_neur_2 = 100
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
W.append(np.random.rand(num_hid_neur, num_hid_neur_2)*0.5)
W.append(np.random.rand(num_hid_neur_2, num_outputs)*0.5)

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

def fwdprop(mat, W):
    z1 = np.dot(mat, W[0])
    a1 = relu(z1)
    z2 = np.dot(a1, W[1])
    a2 = relu(z2)
    z3 = np.dot(a2, W[2])
    yhat = softmax(z3)
    return yhat

def backprop(mat, y, W):

    #fwd prop
    z1 = np.dot(mat, W[0])
    a1 = relu(z1)
    z2 = np.dot(a1, W[1])
    a2 = relu(z2)
    z3 = np.dot(a2, W[2])
    yhat = softmax(z3)

    #bck prop
    delta3 = yhat-y
    dJdw3 = np.dot(a2.T, delta3)

    delta2 = np.dot(delta3, W[2].T)*reluderiv(z2)
    dJdw2 = np.dot(a1.T, delta2)

    delta1 = np.dot(delta2, W[1].T)*reluderiv(z1)
    dJdw1 = np.dot(mat.T, delta1)
    return dJdw1, dJdw2, dJdw3


for i in range(iterations):
    #pick a batch
    samp_index = random.sample(range(0, X_train.shape[0]-1), batch_size)

    X_train_batch = np.zeros((batch_size,28*28))

    #X_train_batch = []
    for index in range(0,len(samp_index)-1):
        X_train_batch[index] = X_train[samp_index[index]]
    # print(X_train_small)
    # print("iter ",i)
    #
    Y_train_batch_mat = np.zeros((batch_size,num_outputs))
    for r in range(batch_size):
        Y_train_batch_mat[r][Y_train[samp_index[r]]] = 1

    #X_train_batch = X_train[batch_size*i:batch_size+batch_size*i]
    #Y_train_batch = Y_train[batch_size*i:batch_size+batch_size*i]

    #Y_train_batch_mat = np.zeros((batch_size,num_outputs))

    #for r in range(batch_size):
        #Y_train_small_mat[r][Y_train_small[r]] = 1

    #print(Y_train_small_mat)
    #print(Y_train_small)

    #do back prop
    dJdw1, dJdw2, dJdw3  = backprop(X_train_batch, Y_train_batch_mat, W)

    #update weights
    W[0] = W[0] - step_size * dJdw1/np.max(np.fabs(dJdw1))
    W[1] = W[1] - step_size * dJdw2/np.max(np.fabs(dJdw2))
    W[2] = W[2] - step_size * dJdw3/np.max(np.fabs(dJdw3))


test_size = 30

res = fwdprop(X_test,W)
print(np.around(res,5))
xout = []
yout = []
for x in range(res.shape[0]):
    xout.append(np.argmax(res[x]))
#print(xout)

for y in Y_test:
    yout.append(y)
#print(yout)

comp = np.matrix(yout)-np.matrix(xout)
comp2 = comp==0

print(np.around(np.sum(comp2)/comp2.shape[1]*100,2))

##f = open('results/results'+datetime.now().strftime("%Y%m%d%H%M%S")+'.txt','w')

#np.set_printoptions(threshold='nan')

##f.write(str(np.around(np.sum(comp2)/comp2.shape[1]*100,2))+"\n")

#f.write(str(W[0]))

#f.write(str(W[1])+"\n")

##f.close()

#print("comp2",comp2)


#print(Y_test[60:test_size+60])
#added a comment
