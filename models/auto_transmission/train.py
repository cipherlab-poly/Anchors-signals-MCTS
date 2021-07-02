import numpy as np
from auto_transmission import AutoTransmission
from sklearn import linear_model
from joblib import dump
from os.path import *

duration = 15
tdelta = 0.5
slen = int(duration/tdelta)
throttles = [0.5]*slen
thetas = [0.]*slen

tlen = 1500
inputs = np.zeros((tlen, slen*2))
outputs = np.zeros(tlen)

t = 0
for _ in range(750):
    at = AutoTransmission(throttles, thetas, tdelta=tdelta)
    at.run()
    inputs[t, :slen] = at.espds
    inputs[t, slen:] = at.vspds
    outputs[t] = 0
    t += 1
for _ in range(250):
    at = AutoTransmission(throttles, thetas, tdelta=tdelta)
    at.run(fault1=True)
    inputs[t, :slen] = at.espds
    inputs[t, slen:] = at.vspds
    outputs[t] = 1
    t += 1
for _ in range(250):        
    at = AutoTransmission(throttles, thetas, tdelta=tdelta)
    at.run(fault2=True)
    inputs[t, :slen] = at.espds
    inputs[t, slen:] = at.vspds
    outputs[t] = 2
    t += 1
for _ in range(250):        
    at = AutoTransmission(throttles, thetas, tdelta=tdelta)
    at.run(fault3=True)
    inputs[t, :slen] = at.espds
    inputs[t, slen:] = at.vspds
    outputs[t] = 3
    t += 1

regr = linear_model.LogisticRegression(max_iter=10000)
regr.fit(inputs, outputs)
filename = dirname(abspath(__file__)) + '/auto_transmission.joblib'
dump(regr, filename)
