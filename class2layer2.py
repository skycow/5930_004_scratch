#Skyler Cowley
#ECE5930-004
#Assignment 3

import numpy as np

def readInData():
    #read in all data
    a = np.loadtxt('classasgntrain1.dat')

    b = a[:,0:2]
    c = np.hstack((b,np.zeros((a.shape[0],1))))

    d = a[:,2:4]
    e = np.hstack((d,np.ones((a.shape[0],1))))

    A = np.vstack((c,e))

    return A

def createData():
    a = readInData()
    np.random.shuffle(a)

    xtrain = a[0:160,0:2]
    ytrain = a[0:160,2].reshape(160,1)
    xtest = a[160:,0:2]
    ytest = a[160:,2].reshape(40,1)

    return [xtrain, ytrain, xtest, ytest]

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

def sigmoid(mat):
    return 1/(1+np.exp(-mat))

def sigmoidderiv(mat):
    return np.exp(-mat)/((1+np.exp(-mat))**2)

def jerror(jyhat, jy):
    #return (1/10)*np.sum(jyhat-jy)**2
    return np.linalg.norm(jyhat-jy,2)/100

def fwdprop(mat, W):
    z1 = np.dot(mat, W[0])
    a1 = relu(z1)
    z2 = np.dot(a1, W[1])
    a2 = relu(z2)
    z3 = np.dot(a2, W[2])
    yhat = sigmoid(z3)
    return yhat

def backprop(mat, backy, W):

    #fwd prop
    z1 = np.dot(mat, W[0])
    a1 = relu(z1)
    z2 = np.dot(a1, W[1])
    a2 = relu(z2)
    z3 = np.dot(a2, W[2])
    yhat = sigmoid(z3)

    #bck prop
    delta3 = yhat-backy
    #delta3 = np.multiply(-(backy-yhat),sigmoidderiv(z2))
    dJdw3 = np.dot(a2.T, delta3)

    delta2 = np.dot(delta3, W[2].T)*reluderiv(z2)
    dJdw2 = np.dot(a1.T, delta2)

    delta1 = np.dot(delta2, W[1].T)*reluderiv(z1)
    dJdw1 = np.dot(mat.T, delta1)
    return dJdw1, dJdw2, dJdw3

batch_size = 160
num_inputs = 2
num_hid_neur = 5
num_hid_neur_2 = 10
num_outputs = 1
step_size = 0.07
iterations = 1
beta = 0.8

a = createData()

X_train = a[0]
Y_train = a[1]

X_test = a[2]
Y_test = a[3]

W = [];

W.append(np.random.rand(num_inputs, num_hid_neur)*0.5)
W.append(np.random.rand(num_hid_neur, num_hid_neur_2)*0.5)
W.append(np.random.rand(num_hid_neur_2, num_outputs)*0.5)

dJdw1p = 0
dJdw2p = 0
dJdw3p = 0

for i in range(0,160,2):
    dJdw1, dJdw2, dJdw3  = backprop(X_train[i:i+1], Y_train[i:i+1], W)

    #update weights
    if (np.max(np.fabs(dJdw2)) != 0 and np.max(np.fabs(dJdw1)) != 0 and np.max(np.fabs(dJdw3)) != 0):
        W[0] = W[0] - (step_size * dJdw1/np.max(np.fabs(dJdw1)) + beta * dJdw1p)
        W[1] = W[1] - (step_size * dJdw2/np.max(np.fabs(dJdw2)) + beta * dJdw2p)
        W[2] = W[2] - (step_size * dJdw3/np.max(np.fabs(dJdw3)) + beta * dJdw3p)
        dJdw1p = (step_size * dJdw1/np.max(np.fabs(dJdw1)) + beta * dJdw1p)
        dJdw2p = (step_size * dJdw2/np.max(np.fabs(dJdw2)) + beta * dJdw2p)
        dJdw3p = (step_size * dJdw3/np.max(np.fabs(dJdw3)) + beta * dJdw3p)


    errorcalc = jerror(fwdprop(X_test,W),Y_test)
    print("MSE: " + str(errorcalc))



#testing
res = fwdprop(X_test,W)
xout = []
for x in range(res.shape[0]):
    xout.append(np.round(res[x]))

comp = np.matrix(Y_test)-np.matrix(xout).reshape(40,1)
comp2 = comp==0
print(np.hstack((Y_test, np.matrix(xout).reshape(40,1))))
print(np.around(np.sum(comp2)/comp2.shape[0]*100,2))
