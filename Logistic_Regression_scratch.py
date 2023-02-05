import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

study_time = np.array([1,2,3,6,8])
pass_flag = np.array([0,0,0,1,1])

test_study_time = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])

# x_{0} = 1, x_{1} = study_time / y = pass_flag: 0=Fail, 1=Pass

def sigmoid(x): # sigmoid function
    z = 1/(1+np.exp(-x))
    return z

def dev_sigmoid(x): # dev_sigmoid function
    z = sigmoid(x)*(1-sigmoid(x))
    return z

def pred(x,w): # prediction
    #print('x and w:', x, w)
    z = w[0] + w[1]*x
    #z = sigmoid(w[0] + w[1]*x)
    return z

def cost(x,y,w): # cross entropy : binary classification cost function
    z = np.mean(-y*np.log(sigmoid(pred(x,w)))-(1-y)*np.log(1-sigmoid(pred(x,w))))
    return z

def dev_cost(x,y,w):
    z0 = np.mean(sigmoid(pred(x,w))-y) # x_{0} = 1
    z1 = np.mean(x*(sigmoid(pred(x,w))-y))
    return np.array([z0,z1])

#w = np.random.random(2) # w_{0}, w_{1}. y = w_{0}*x_{0} + w_{1}*x_{1}
w = np.zeros(2) # w_{0}, w_{1}. y = w_{0}*x_{0} + w_{1}*x_{1}
J = cost(study_time, pass_flag, w)
print('Intial J:', J)
J_prev = 0
epsilon = 1e-3
lr = 1 # learning rate
num_iter = 0

print('partial J partial w_{j}:\n', dev_cost(study_time,pass_flag,w))
print('Shape of partial J partial w_{j}:\n', dev_cost(study_time,pass_flag,w).shape)
while np.abs((J-J_prev)/J) > epsilon:
    num_iter += 1
    w = w - lr*dev_cost(study_time,pass_flag,w) # gradient descend
    print('weight at iter :' + str(num_iter), w)
    J_prev = J
    J = cost(study_time,pass_flag,w)
    print('J at iter :' + str(num_iter), J)

test_study_time = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])

pred_prob = sigmoid(pred(test_study_time, w))
pred_flag = np.copy(pred_prob)
pred_flag[pred_flag>=0.5] = 1
pred_flag[pred_flag<0.5] = 0

print('Students who studied: \n', test_study_time)
print('are predicted to pass the test :\n', pred_flag)
print('(probability):\n', pred_prob)
