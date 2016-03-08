import numpy as np
from scipy.optimize import *
import math

class EventFct():
    def __init__(self, tmax, t0 = 0, cst = 1., tau = 1.):
        self.tmax= tmax
        if t0 <= 0 or t0 >= tmax :
            self.t0 = tmax/2
        else:
            self.t0 = t0
        self.tau = tau
        self.values = np.zeros(tmax)
        self.cst = cst
        
    def createFct(self):
        self.t0 = round(self.t0, 3)
        self.values[:math.ceil(self.t0)] = 0.
        self.values[math.ceil(self.t0):] = self.cst * np.exp (-1 * np.arange(self.tmax-math.ceil(self.t0)) * self.tau)
        
    def setParams(self, t0 = None, cst = None, tau = None):
        if t0 != None:
            self.t0 = t0
        if cst != None:
            self.cst = cst
        if tau != None:
            self.tau = tau
    
class FitEvent():
    def __init__(self, data, cst, t0 = 0, tau = 1):
        self.data = data
        self.tmax = len(data)
        self.fitFct = EventFct(self.tmax, t0 = t0, tau = tau, cst = cst)
        
    def errorfunction(self, params, errors):
        f = EventFct(self.tmax, tau = params[0], cst = self.fitFct.cst, t0 = self.fitFct.t0)
        f.createFct()
        if params[0] < 0:
            params[0] = 0.
        err = (f.values - self.data)**2
        if len(params[params<0]) != 0:
            err *= 2
        errors.append(err)
        return np.mean(err)*10000
        
    def fit(self):
        errors = []
        params = np.array([self.fitFct.tau])
        out = fmin(self.errorfunction, params, full_output = True, disp = False, args = (errors,))
        self.fitFct.setParams(tau = out[0][0])
        self.fitFct.createFct()
        return out[0], out[1]
           
class Trajectory():
    def __init__(self, data, window, mincenter, minpara, varmin, hthreshold=0.8, lthreshold=0.1):
        self.data = data
        self.lengthSeq = len(data)
        self.hthreshold = hthreshold
        self.lthreshold = lthreshold
        #print "low threshold :", self.lthreshold
        self.window = window
        self.mincenter = mincenter
        self.minpara = minpara
        self.varmin = varmin
        self.tau = -1.
        self.max_ = -1
        self.maxis = []
        self.f1=[]
        self.f2=[]
        self.f3=[]
        self.fgeo=[]
     
    def append(self, obj):
        if type(obj) == list or type(obj) == np.array:
            self.data.extend(obj)
            self.lengthSeq += len(obj)
        else:
            self.data.append(obj)
            self.lengthSeq += 1
        
    def findMax(self):
        self.maxis = np.arange(self.lengthSeq)[np.array(self.data) > self.hthreshold]

     
    def rescale(self):
        self.data = np.array(self.data)
        self.data /= np.max(self.data)
        
    def decision(self, rangeTau = [0., 1.5], th_amplitude = 10, percent_low_before = 0.5):
        self.findMax()
        maxTrue = -1
        if len(self.maxis) > 0:
            for m in self.maxis:
                amplitude = self.data[m]/np.mean(self.data[:m])
                dismiss = np.mean(self.data[:m] > self.lthreshold)
                t, err, p, fit = self.testEvent(m)
                resp = (p > rangeTau[0] and p < rangeTau[1])
                if dismiss < percent_low_before and amplitude >= th_amplitude and m > maxTrue and resp:
                    maxTrue = m
                    self.tau = p
                    break
                elif dismiss > percent_low_before:
                    ## dismiss is increasing with time
                    break
        self.max_ = maxTrue
        return (maxTrue != -1)

    def findMax_lyb(self):
        self.maxis = np.nonzero(self.data > self.hthreshold)[0]
        self.maxis = [x for x in self.maxis if x > len(self.data)-self.window-1 and x < len(self.data)-self.window/2]

    def decision_lyb_off(self):# offline detection of cell division
        # use product of 3 features's gradient to find global maxia and then decide if it is division by standard
        # derivation and the threshold from large-scale experiment
        self.f1=np.gradient(self.data)
        self.f2=np.gradient((self.varmin-np.min(self.varmin))/(np.max(self.varmin)-np.min(self.varmin)))
        self.f3=np.gradient( (self.mincenter-np.min(self.mincenter))/(np.max(self.mincenter)-np.min(self.mincenter)))

        gemean = self.f1*self.f2*self.f3
        gemean = np.concatenate([[gemean[0]],gemean[:-1]])
        self.fgeo = gemean
        self.f1 = np.concatenate([[self.f1[0]],self.f1[:-1]])
        self.f2 = np.concatenate([[self.f2[0]],self.f2[:-1]])
        self.f3 = np.concatenate([[self.f3[0]],self.f3[:-1]])

        mv_i = np.argmax(gemean)
        mv=gemean[mv_i]
        prev = gemean[0:mv_i-5]
        prev.sort()
        prev = prev[0:int(np.floor(len(prev)*0.6))]
        prevmean = np.mean(prev)
        prevstd = np.std(prev)
        numstd= np.floor((mv - prevmean)/prevstd)

        predef=100
        re = -1
        if numstd > predef:
            re=mv_i

        return re, mv_i, numstd

    def decision_lyb_off_testeachfeature(self):# offline detection of cell division
        # use product of 3 features's gradient to find global maxia and then decide if it is division by standard
        # derivation and the threshold from large-scale experiment
        self.f1=np.gradient(self.data)
        self.f2=np.gradient((self.varmin-np.min(self.varmin))/(np.max(self.varmin)-np.min(self.varmin)))
        self.f3=np.gradient( (self.mincenter-np.min(self.mincenter))/(np.max(self.mincenter)-np.min(self.mincenter)))

        gemean = self.f1*self.f2*self.f3
        gemean = np.concatenate([[gemean[0]],gemean[:-1]])
        self.fgeo = gemean
        self.f1 = np.concatenate([[self.f1[0]],self.f1[:-1]])
        self.f2 = np.concatenate([[self.f2[0]],self.f2[:-1]])
        self.f3 = np.concatenate([[self.f3[0]],self.f3[:-1]])

        #f1
        final = self.f1
        mv_i = np.argmax(final)
        mv=final[mv_i]
        prev = final[0:mv_i-5]
        prev.sort()
        prev = prev[0:int(np.floor(len(prev)*0.6))]
        prevmean = np.mean(prev)
        prevstd = np.std(prev)
        numstd= np.floor((mv - prevmean)/prevstd)

        predef=100
        re1 = -1
        if numstd > predef:
            re1=mv_i
        #f2
        final = self.f2
        mv_i = np.argmax(final)
        mv=final[mv_i]
        prev = final[0:mv_i-5]
        prev.sort()
        prev = prev[0:int(np.floor(len(prev)*0.6))]
        prevmean = np.mean(prev)
        prevstd = np.std(prev)
        numstd= np.floor((mv - prevmean)/prevstd)

        predef=100
        re2 = -1
        if numstd > predef:
            re2=mv_i
        #f3
        final = self.f3
        mv_i = np.argmax(final)
        mv=final[mv_i]
        prev = final[0:mv_i-5]
        prev.sort()
        prev = prev[0:int(np.floor(len(prev)*0.6))]
        prevmean = np.mean(prev)
        prevstd = np.std(prev)
        numstd= np.floor((mv - prevmean)/prevstd)

        predef=100
        re3 = -1
        if numstd > predef:
            re3=mv_i
        #f1 and f2
        final = self.f1*self.f2
        mv_i = np.argmax(final)
        mv=final[mv_i]
        prev = final[0:mv_i-5]
        prev.sort()
        prev = prev[0:int(np.floor(len(prev)*0.6))]
        prevmean = np.mean(prev)
        prevstd = np.std(prev)
        numstd= np.floor((mv - prevmean)/prevstd)

        predef=100
        re12 = -1
        if numstd > predef:
            re12=mv_i
        #f1 and f3
        final = self.f1*self.f3
        mv_i = np.argmax(final)
        mv=final[mv_i]
        prev = final[0:mv_i-5]
        prev.sort()
        prev = prev[0:int(np.floor(len(prev)*0.6))]
        prevmean = np.mean(prev)
        prevstd = np.std(prev)
        numstd= np.floor((mv - prevmean)/prevstd)

        predef=100
        re13 = -1
        if numstd > predef:
            re13=mv_i
        #f2 and f3
        final = self.f2*self.f3
        mv_i = np.argmax(final)
        mv=final[mv_i]
        prev = final[0:mv_i-5]
        prev.sort()
        prev = prev[0:int(np.floor(len(prev)*0.6))]
        prevmean = np.mean(prev)
        prevstd = np.std(prev)
        numstd= np.floor((mv - prevmean)/prevstd)

        predef=100
        re23 = -1
        if numstd > predef:
            re23=mv_i
        # all feature
        mv_i = np.argmax(gemean)
        mv=gemean[mv_i]
        prev = gemean[0:mv_i-5]
        prev.sort()
        prev = prev[0:int(np.floor(len(prev)*0.6))]
        prevmean = np.mean(prev)
        prevstd = np.std(prev)
        numstd= np.floor((mv - prevmean)/prevstd)

        predef=100
        re = -1
        if numstd > predef:
            re=mv_i
        #
        reall=[re1,re2,re3,re12,re13,re23,re]
        return reall



    def extractEvent(self, t):
        event = self.data[max(0,t - self.window): min(self.lengthSeq,t + self.window)]
        return event

    def testEvent(self, t = None):
        if t == None:
            t = self.max_
        extractData = self.extractEvent(t)
        t0 = np.argmax(extractData)
        cst = np.max(extractData)
        tau = np.log(cst/extractData[-1])/(t+self.window)
        fitEvent = FitEvent(extractData, cst, t0 = t0, tau = tau)
        params, err = fitEvent.fit()
        return t, err, params[0], fitEvent.fitFct.values
