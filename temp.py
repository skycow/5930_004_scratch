from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random


mnist = input_data.read_data_sets("/tmp/data/")

X_train = mnist.train.images
Y_train = mnist.train.labels.astype("int")

X_test = mnist.test.images
Y_test = mnist.test.labels.astype("int")

batch_size = 100
num_mats = 28*28
num_hid_neur = 300
num_outputs = 10

X_train_small = X_train[0:batch_size]
Y_train_small = Y_train[0:batch_size]

Y_train_small_mat = np.zeros((batch_size,num_outputs))

for r in range(batch_size):
    Y_train_small_mat[r][Y_train_small[r]] = 1

W = [];

W.append(np.random.rand(num_mats, num_hid_neur)*0.5*2)
W.append(np.random.rand(num_hid_neur, num_outputs)*0.5*2)

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

def softmaxderiv(mat):
    temp1 = softmax(mat)
    return np.multiply(temp1,(1 - temp1))

def sigmoid(mat):
    return 1/(1+np.exp(-mat))

def sigmoidderiv(mat):
    return np.exp(-mat)/((1+np.exp(-mat))**2)

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
    #a1 = a1*(1/np.max(a1))
    z2 = np.dot(a1, W[1])
    #z2 = z2*(1/np.max(z2))
    yhat = softmax(z2)
    return yhat

def cel(yhat, y):
    celsum = 0
    for n in range(yhat.shape[0]):
        for i in range(yhat.shape[1]):
            celsum = celsum + y[n][i]*np.log(yhat[n][i])

def backprop(mat, y, W):

    #fwd prop
    z1 = np.dot(mat, W[0])
    a1 = relu(z1)
    #a1 = a1*(1/np.max(a1))
    z2 = np.dot(a1, W[1])
    #z2 = z2*(1/np.max(z2))
    yhat = softmax(z2)

    #bck prop
    #temp = softmaxderiv(z2)
    #delta3 = np.multiply(-(y-yhat),temp)
    delta3 = yhat-y
    dJdw2 = np.dot(a1.T, delta3)#*(1/a1.shape[0])

    delta2 = np.dot(delta3, W[1].T)*reluderiv(z1)
    dJdw1 = np.dot(mat.T, delta2)#*(1/mat.shape[0])
    return dJdw1, dJdw2

#res = fwdprop(X_train_small, W)
for i in range(500):
    #pick a batch
    # samp_index = random.sample(range(0, X_train.shape[0]-1), batch_size)
    # print(samp_index)
    #
    # X_train_small = []
    # for index in samp_index:
    #     X_train_small.append(X_train[index])
    # print(X_train_small)
    # print("iter ",i)
    #
    # Y_train_small_mat = np.zeros((batch_size,num_outputs))
    # for r in range(batch_size):
    #     Y_train_small_mat[r][Y_train[samp_index[r]]] = 1
    X_train_small = X_train[batch_size*i:batch_size+batch_size*i]
    Y_train_small = Y_train[batch_size*i:batch_size+batch_size*i]

    Y_train_small_mat = np.zeros((batch_size,num_outputs))

    for r in range(batch_size):
        Y_train_small_mat[r][Y_train_small[r]] = 1

    #print(Y_train_small_mat)
    #print(Y_train_small)

    #do back prop
    dJdw1, dJdw2  = backprop(X_train_small, Y_train_small_mat, W)

    #update weights
    W[0] = W[0] - 0.4 * dJdw1/np.max(np.fabs(dJdw1))
    W[1] = W[1] - 0.4 * dJdw2/np.max(np.fabs(dJdw2))

    #W[0] = W[0] + np.fabs(np.min(W[0]))
    #W[1] = W[1] + np.fabs(np.min(W[1]))
    #W[0] = W[0] / np.max(np.fabs(W[0]))
    #W[1] = W[1] / np.max(np.fabs(W[1]))

test_size = 30

res = fwdprop(X_test[60:test_size+60],W)
print(np.around(res,5))
xout = []
yout = []
for x in range(res.shape[0]):
    xout.append(np.argmax(res[x]))
print(xout)

for y in Y_test[60:test_size+60]:
    yout.append(y)
print(yout)
print(Y_test[60:test_size+60])
