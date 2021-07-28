import numpy as np
from ..auto_transmission import AutoTransmission
from sklearn import linear_model
from joblib import dump
import os.path

class Train:
    def __init__(self, slen, tdelta):
        self.tdelta = tdelta
        self.slen = slen
        self.throttles = [0.5]*slen
        self.thetas = [0.]*slen
        self.inputs = np.zeros((2000, slen*2))
        self.outputs = np.zeros(2000)

    def train(self):
        t = 0
        for _ in range(1000):
            at = AutoTransmission(self.throttles, self.thetas, self.tdelta)
            at.run()
            self.inputs[t, :self.slen] = at.espds
            self.inputs[t, self.slen:] = at.vspds
            self.outputs[t] = 0
            t += 1
        for _ in range(500):        
            at = AutoTransmission(self.throttles, self.thetas, self.tdelta)
            at.run(fault=1)
            self.inputs[t, :self.slen] = at.espds
            self.inputs[t, self.slen:] = at.vspds
            self.outputs[t] = 1
            t += 1
        for _ in range(500):        
            at = AutoTransmission(self.throttles, self.thetas, self.tdelta)
            at.run(fault=2)
            self.inputs[t, :self.slen] = at.espds
            self.inputs[t, self.slen:] = at.vspds
            self.outputs[t] = 2
            t += 1

        regr = linear_model.LogisticRegression(max_iter=10000)
        regr.fit(self.inputs, self.outputs)
        filename = os.path.dirname(os.path.abspath(__file__))
        filename += '/autotransmission.joblib'
        dump(regr, filename)
