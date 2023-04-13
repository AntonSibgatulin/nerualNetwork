import random
import numpy as np

import math
#import loadData
from include import *
import pandas as pd
from indicators import *

import matplotlib.pyplot as plt

prices = pd.read_csv("data.csv")

data = []
y_true = []






#COUNT_CANDLES = 12
COUNT_OF_POINT = 0.003

#INPUT_DIM = COUNT_CANDLES

OUT_DIM =3

H_DIM = 12

ALPHA = 0.0001
NUM_EPOCHS = 1000

PREPARE = 4
SIZE_TEST = PREPARE*500
DELTA_PROFIT = 80


trading = []
loss_arr = []










INPUT_DIM = 4

cci = CCI(20,prices)
stoh = stochastic_oscillator(prices, k_period=14, d_period=6)
rsi = RSI(18,prices)
sma = SMA(20,prices)
macd=calculate_macd(prices,12,26,9)

for i in range(50,len(cci)):
    #small_arr = np.array([cci[i],stoh['%K'][i],((sma[i]-prices['Close'][i])/Points),macd['macd'][i]])
    #small_arr = np.array([[ round(cci[i]),round(stoh['%D'][i]),math.floor((sma[i]-prices['Close'][i])/Points),macd['macd'][i],macd['signal'][i],rsi[i] ] ])
    small_arr = np.array([[ normales(round(cci[i]),150,-150), normales(round(stoh['%D'][i]),100,0),
                            normales(math.floor(sma[i] - prices['Close'][i]) / Points,50,-50),
                           normales(macd['macd'][i],0.005,-0.005) ]])

    data.append(small_arr)
    max,min = checkMaxMinDt(prices['Open'][i],prices['Close'][i:i+PREPARE])
    #print(small_arr)
    #exit(0)
    tru = 1
    if max>min:
        if max>DELTA_PROFIT:
            tru = 2
    elif min>max:
        if min > DELTA_PROFIT:
            tru = 0
    else:
        tru = 1

    y_true.append(tru)

#ar = prices['Close'][len(prices['Close']):len(prices['Close'])+3]
#print(ar)








W1 = np.random.rand(INPUT_DIM,H_DIM)
b1 = np.random.rand(1,H_DIM)
W2 = np.random.rand(H_DIM,OUT_DIM)
b2 = np.random.rand(1,OUT_DIM)

W1 = (W1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
b1 = (b1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
W2 = (W2 - 0.5) * 2 * np.sqrt(1/H_DIM)
b2 = (b2 - 0.5) * 2 * np.sqrt(1/H_DIM)













for ep in range(NUM_EPOCHS):
    for i in range(len(data)):
        # ALPHA = 1/(random.randrange(10,10000))
        # x = np.random.rand(1,INPUT_DIM)
        # y = random.randint(0,OUT_DIM -1)
        x = data[i]
        y = y_true[i]
        # print(x,y)
        t1 = x @ W1 + b1
        h1 = relu(t1)
        t2 = h1 @ W2 + b2
        z = softmax(t2)
        E = sparse_cross_entropy(z, y)
        y_full = to_full(y, OUT_DIM)
        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = dE_dt2
        dE_dh1 = dE_dt2 @ W2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = dE_dt1

        W1 = W1 - ALPHA * dE_dW1
        b1 = b1 - ALPHA * dE_db1

        W2 = W2 - ALPHA * dE_dW2
        b2 = b2 - ALPHA * dE_db2
        loss_arr.append(E)
        trading.append(np.argmax(z))
        if ep % 100 == 0 and i % 100 == 0:
            print(ep, E, np.argmax(z), y)


def predict(x):
    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax(t2)
    return z




def calc_accuracy():
    correct = 0
    for i in range(len(data)):
        x = data[i]
        y = y_true[i]
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    acc = correct / len(data)
    return acc



accuracy = calc_accuracy()
print("Accuracy:",accuracy)

'''
probs = predict(np.array([array_end]))
pred_class = np.argmax(probs)
print(pred_class)
'''
#print(W1)
#print(W2)
answer = []
true_data = []
count_accept_answer = 0
count_unaccept_answer = 0
count = 0
start = int(len(data)-SIZE_TEST)
if start< 0 :
    start = 0
for i in range(start,int(len(data)),1):
    x = data[i]
    z = predict(x)
    y_pred = np.argmax(z)
    if y_true[i] == y_pred and y_pred!=1:
        count_accept_answer+=1
    if y_true[i]!= y_pred and y_pred!= 1:
        count_unaccept_answer+=1
    count+=1
    answer.append(y_pred)
    true_data.append(y_true[i])
#plt.subplot (1, 1, 1)
plt.plot(answer,color="red")
plt.plot(true_data,color="blue")

print(count_accept_answer,count_unaccept_answer,count)
#plt.subplot (1, 1, 2)

plt.show()



plt.plot(loss_arr)
plt.show()

