import numpy as np
import math
Points = 0.0001
def sigmoid(x):
    # Наша функция активации: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def reLu(input):
    return np.maximum(0,input)

def mes_loss(y_true,y_pred):
    return ((y_true-y_pred)**2).mean()

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def softmax(t):
    out = np.exp(t)
    return out/np.sum(out)

def sparse_cross_entropy(z,y):
    return -np.log(z[0,y])

def to_full(y,num_classes):
    y_full = np.zeros((1,num_classes))
    y_full[0,y] = 1
    return y_full

def reLU_deriv(t):
    return (t>=0).astype(float)


def checkMaxMinDt(open,arr):
    max = open
    min = open
    for i in arr:
        if i > max:
            max = i
        if i < min:
            min = i
    return math.floor((max-open)/Points),math.floor((open - min)/Points)



def relu(t):
    return np.maximum(t, 0)

def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)

def softmax_batch(t):
    out = np.exp(t)
    return out / np.sum(out, axis=1, keepdims=True)

def sparse_cross_entropy(z, y):
    return -np.log(z[0, y])

def sparse_cross_entropy_batch(z, y):
    return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))

def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full

def to_full_batch(y, num_classes):
    y_full = np.zeros((len(y), num_classes))
    for j, yj in enumerate(y):
        y_full[j, yj] = 1
    return y_full

def relu_deriv(t):
    return (t >= 0).astype(float)

def normales(x,x_max,x_min):
    return float((x -x_min) / (x_max - x_min))