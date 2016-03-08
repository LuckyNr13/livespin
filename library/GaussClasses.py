import matplotlib
import pylab
from random import randint
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import *
import scipy
import heapq
from time import time
from scipy import linalg, mat, dot

import utilities

''' GaussianObject, class to model a gaussian.
   Parameters : - nComponents
                - params : each gaussian has 6 parameters (weight, centerX, centerY, sigmaX, sigmaY, rotation angle). It is a 1D vector.
                - thresholds : a high for sigma X and Y, a low for weight. For the weight it can be fix by the user or calculated with the histogram.
    Methods : - to set or update params (updateMoments)
              - get area
              - create the gaussian function to fit to the image : calculate all the values for a grid of coordinates (createGaussian)

              - 2 methods to visualize results : display_params, drawObject
    '''


class GaussianObject():
    colors = ['r', 'c', 'y', 'g', 'm']

    def __init__(self, nComponents, params=None, th_sigma_high=12., th_weight_low=40.):
        self.nComponents = nComponents
        self.params = np.zeros(6 * self.nComponents, dtype=np.float)
        self.th_sigma_high = th_sigma_high
        self.th_weight_low = th_weight_low
        if params != None:
            if len(params) / 6 != nComponents:
                raise IndexError("Not coherent : nComponents and number of parameters.")
            self.updateMoments(params)


    def updateMoments(self, params):
        params = np.array(params)
        self.params = params
        if len((np.abs(params[3::6]) - self.th_sigma_high)[params[3::6] > self.th_sigma_high]) != 0 or len(
                (np.abs(params[4::6]) - self.th_sigma_high)[params[4::6] > self.th_sigma_high]) != 0:
            for i in range(self.nComponents):
                self.params[6 * i + 3] = min(self.th_sigma_high, np.abs(params[6 * i + 3]))
                self.params[6 * i + 4] = min(self.th_sigma_high, np.abs(params[6 * i + 4]))
        if len(self.params[::6][np.abs(self.params[::6]) < self.th_weight_low]) != 0:
            for i in range(self.nComponents):
                self.params[6 * i] = max(self.th_weight_low, np.abs(params[6 * i]))

    def updateMoments_new(self, params):
        params = np.array(params)
        self.params = params
        self.params[5::6] %= (np.pi / 2.0 * self.params[5::6] / abs(self.params[5::6]))
        self.params[0::6] = self.params[0::6] ** 2

    def getArea(self):
        return np.pi * self.params[3::6] * self.params[4::6]



    def createGaussian(self, p):
        def rotgauss(x, y):
            f = np.zeros(x.shape)
            for i in xrange(self.nComponents):

                exposant = np.exp(-1. / (2. * (1 - p[6 * i + 5] ** 2)) * (
                    (x - p[6 * i + 1]) ** 2 / p[6 * i + 3] ** 2 - 2. * p[6 * i + 5] * (x - p[6 * i + 1]) /
                    p[6 * i + 3] * (
                        y - p[6 * i + 2]) / p[6 * i + 4] + (y - p[6 * i + 2]) ** 2 / p[6 * i + 4] ** 2))
                fi = p[6 * i] ** 2 * exposant
                f += fi
            return f

        return rotgauss

    def changeCenters(self, xExtract, yExtract):
        self.params[1::6] += xExtract
        self.params[2::6] += yExtract
        return True



    def draw2D(self):
        for i in xrange(self.nComponents):
            xeq = lambda t: self.params[6 * i + 3] * np.cos(t) * np.cos(self.params[6 * i + 5]) + self.params[
                                                                                                      6 * i + 4] * np.sin(
                t) * np.sin(self.params[6 * i + 5]) + self.params[6 * i + 1]
            yeq = lambda t: - self.params[6 * i + 3] * np.cos(t) * np.sin(self.params[6 * i + 5]) + self.params[
                                                                                                        6 * i + 4] * np.sin(
                t) * np.cos(self.params[6 * i + 5]) + self.params[6 * i + 2]
            t = np.linspace(0, 2 * np.pi, 100)
            x = xeq(t)
            y = yeq(t)
            pylab.scatter(self.params[6 * i + 2], self.params[6 * i + 1], color='k')
            pylab.plot(y.astype(int), x.astype(int), self.colors[i] + '-')

    def draw2D_new(self):
        for i in xrange(self.nComponents):
            k1 = np.array([[self.params[6 * i + 3] ** 2, self.params[6 * i + 3] * self.params[6 * i + 4] * self.params[6 * i + 5]],
                           [self.params[6 * i + 3] * self.params[6 * i + 4] * self.params[6 * i + 5], self.params[6 * i + 4] ** 2]])
            w1, v1 = np.linalg.eig(k1)
            idx = w1.argsort()
            w1 = w1[idx]
            v1 = v1[:, idx]
            angle=-(np.arctan(v1[1][1]/v1[0][1]))+np.pi#x+2*(pi/4-x)+pi/2#since in the image X and Y are inverted, so need to minus 90 degree and flip around pi/4

            w2 = np.zeros((1 , 2))
            w2[0,1] = np.sqrt(2)*np.max([self.params[6 * i + 3], self.params[6 * i + 4]])
            w2[0,0] = w2[0,1]*w1[0]/w1[1]

            xeq = lambda t: w2[0,1] * np.cos(t) * np.cos(angle) + w2[0,0] * np.sin(
                t) * np.sin(angle) + self.params[6 * i + 1]
            yeq = lambda t: - w2[0,1] * np.cos(t) * np.sin(angle) + w2[0,0] * np.sin(
                t) * np.cos(angle) + self.params[6 * i + 2]
            t = np.linspace(0, 2 * np.pi, 100)
            x = xeq(t)
            y = yeq(t)
            pylab.scatter(self.params[6 * i + 2], self.params[6 * i +1], color='k')
            pylab.plot(y.astype(int), x.astype(int), self.colors[i] + '-')


    def display_params(self):
        for i in xrange(0, len(self.params), 6):
            print "GAUSSIAN {0} : w {1:.3f}, cx {2:.1f}, cy {3:.1f}, sigmax {4:.1f}, sigmay {5:.1f}, rotation {6:.1f}, surface {7:.1f}".format(
                i / 6 + 1, self.params[i] ** 2, self.params[i + 1], self.params[i + 2], np.abs(self.params[i + 3]),
                np.abs(self.params[i + 4]), self.params[i + 5],
                (np.pi * np.abs(self.params[i + 3]) * np.abs(self.params[i + 4])))


    def switchcomp2(self, prev_para, prev_para2, curr_para):
        bet = 0
        ent = 6
        if prev_para.size == 12:
            distrec = np.finfo(np.float64).max
            indrec = [-1, -1]
            for i in xrange(2):
                for j in xrange(2):
                    if i == j:
                        continue
                    tempdist = np.linalg.norm(prev_para[bet:ent] - curr_para[6 * i + bet:6 * i + ent]) + np.linalg.norm(
                        prev_para[6 + bet:6 + ent] - curr_para[6 * j + bet:6 * j + ent]) \
                               + np.linalg.norm(
                        prev_para2[bet:ent] - curr_para[6 * i + bet:6 * i + ent]) + np.linalg.norm(
                        prev_para2[6 + bet:6 + ent] - curr_para[6 * j + bet:6 * j + ent])

                    if tempdist < distrec:
                        distrec = tempdist
                        indrec = [i, j]
            fin_para = np.concatenate(
                (curr_para[6 * indrec[0]:6 * indrec[0] + 6], curr_para[6 * indrec[1]:6 * indrec[1] + 6]))
        return fin_para

    def switchcomp3(self, prev_para, curr_para2, curr_para):
        bet = 0
        ent = 6
        if prev_para.size == 18:
            distrec = np.finfo(np.float64).max
            indrec = [-1, -1, -1]
            for i in xrange(3):
                for j in xrange(3):
                    for m in xrange(3):
                        if i == j or i == m or j == m:
                            continue
                        tempdist = np.linalg.norm(
                            prev_para[bet:ent] - curr_para[6 * i + bet:6 * i + ent]) + np.linalg.norm(
                            prev_para[6 + bet:6 + ent] - curr_para[6 * j + bet:6 * j + ent]) + np.linalg.norm(
                            prev_para[6 + 6 + bet:6 + 6 + ent] - curr_para[6 * m + bet:6 * m + ent]) \
                                   + np.linalg.norm(curr_para2[1:3] - curr_para[6 * i + 1:6 * i + 3]) + np.linalg.norm(
                            curr_para2[6 + 1:6 + 3] - curr_para[6 * j + 1:6 * j + 3])
                        if tempdist < distrec:
                            distrec = tempdist
                            indrec = [i, j, m]
            fin_para = np.concatenate((
                curr_para[6 * indrec[0]:6 * indrec[0] + 6], curr_para[6 * indrec[1]:6 * indrec[1] + 6],
                curr_para[6 * indrec[2]:6 * indrec[2] + 6]))
            distrec = np.finfo(np.float64).max
            for i in xrange(3):
                tempdist = np.linalg.norm(curr_para2[1:3] - fin_para[6 * i + 1:6 * i + 3]) + np.linalg.norm(prev_para[1:3] - fin_para[6 * i + 1:6 * i + 3])
                if tempdist < distrec:
                    distrec = tempdist
                    indrec = [i]
                    reci = i
            distrec = np.finfo(np.float64).max
            for i in xrange(3):
                if i == reci:
                    continue
                tempdist = np.linalg.norm(curr_para2[6 + 1:6 + 3] - fin_para[6 * i + 1:6 * i + 3]) + np.linalg.norm(
                    prev_para[6 + 1:6 + 3] - fin_para[6 * i + 1:6 * i + 3])
                if tempdist < distrec:
                    distrec = tempdist
                    indrec = [reci, i]
            indrec = np.concatenate((indrec, [3 - sum(indrec)]))
            fin_para2 = np.concatenate((
                fin_para[6 * indrec[0]:6 * indrec[0] + 6], fin_para[6 * indrec[1]:6 * indrec[1] + 6],
                fin_para[6 * indrec[2]:6 * indrec[2] + 6]))

        return fin_para2


class GaussianForFit(utilities.ExtractImage, GaussianObject):
    ''' Prolongation of the previous class GaussianObject.
    Take into account the image : important for the initialization of parameters (weight, centers)
    '''

    def __init__(self, image, nComponents, mask=None, params=None, th_sigma_high=12., th_weight_low=40.):
        utilities.ExtractImage.__init__(self, image)
        GaussianObject.__init__(self, nComponents, params=params, th_sigma_high=th_sigma_high,
                                th_weight_low=th_weight_low)
        self.mask = mask
        if params == None:
            self.gaussian = np.zeros(image.shape)
        else:
            dx, dy = np.indices(self.shape)
            self.gaussian = self.createGaussian(params)(dx, dy)
        if self.mask != None:
            self.distanceMap = self.getDistanceMap()
        else:
            self.distanceMap = None
        self.error = -1.
        self.varmin = []
        self.center23=[]
        self.param23=[]
        self.inipara=[]

    def initMoments(self, centersX, centersY, parai):
        nCenters = len(centersX)

        newCentersX, newCentersY = [], []
        if self.nComponents < nCenters:
            if len(parai) < 1:
                valCenters = self.image[centersX, centersY]

                eudist = np.ones(nCenters)
                for i in range(nCenters):
                    for j in range(nCenters):
                        if j == i:
                            continue
                        eudist[i] += np.linalg.norm(
                            np.array([centersX[i], centersY[i]]) - np.array([centersX[j], centersY[j]]))
                valCenters *= eudist

                chosenCenters = sorted(range(len(valCenters)), key=lambda x: valCenters[x])[-self.nComponents:]
                newCentersX = centersX[chosenCenters]
                newCentersY = centersY[chosenCenters]
            else:
                prevcenter=[]
                for i in xrange(self.nComponents):
                    prevcenter.append(parai[len(parai) - 1][self.nComponents - 2].params[i*6+1:i*6+3])

                recind = np.ones(self.nComponents)*-1
                #
                for k in range(self.nComponents):
                    maxeudist = np.finfo(np.float64).max
                    for i in xrange(nCenters):
                        if i in recind:
                            continue
                        eudist=0
                        for j in range(self.nComponents):
                            eudist += np.linalg.norm(
                                np.array(prevcenter[j][0],prevcenter[j][1]) - np.array([centersX[i], centersY[i]]))
                        eudist /= self.nComponents
                        eudist2 = 0
                        for s in range(k):
                            eudist2 += np.linalg.norm(
                                np.array([centersX[recind[s]], centersY[recind[s]]]) - np.array([centersX[i], centersY[i]]))/k
                        if eudist-eudist2 < maxeudist:
                            cct = i
                            maxeudist = eudist-eudist2
                    recind[k] =cct
                recind = recind.astype(int)
                newCentersX = centersX[recind]
                newCentersY = centersY[recind]

        elif self.nComponents == nCenters:
            newCentersX = centersX
            newCentersY = centersY
        else:
            if len(parai) < 1:
                coordinatesX_mask, coordinatesY_mask = np.indices(self.shape)
                coordinatesX_mask = coordinatesX_mask[self.mask == 1]
                coordinatesY_mask = coordinatesY_mask[self.mask == 1]
                chosenCenters = np.random.randint(0, len(coordinatesX_mask), self.nComponents - nCenters)
                newCentersX = np.append(centersX, coordinatesX_mask[chosenCenters])
                newCentersY = np.append(centersY, coordinatesY_mask[chosenCenters])
                #print "init Centers", newCentersX, newCentersY

            else:
                left=np.zeros([1, self.nComponents])
                leftvalue=np.zeros([1, self.nComponents])
                recind = []
                prevcenter = []
                for i in xrange(self.nComponents):
                    prevcenter.append(parai[len(parai) - 1][self.nComponents - 2].params[i*6+1:i*6+3])

                for j in xrange(self.nComponents):
                    maxrec = 10000
                    for i in xrange(nCenters):
                        if i in recind:
                            continue
                        eudist = np.linalg.norm(np.array(prevcenter[j][:]) - np.array([centersX[i], centersY[i]]))
                        if eudist <maxrec:
                            leftvalue[0][j] = eudist
                tv = np.sort(leftvalue*-1)*-1
                tvi = tv[0][self.nComponents - nCenters-1]
                tts = np.nonzero(leftvalue[0] >= tvi)[0]
                leftfromprev = []
                newCentersX = centersX
                newCentersY = centersY
                for i in xrange(len(tts)):
                    leftfromprev.append(prevcenter[tts[i]][:])
                for i in xrange(len(leftfromprev)):
                    tep = np.ones(self.shape)
                    tep[leftfromprev[i][0],leftfromprev[i][1]] = 0
                    distmap = scipy.ndimage.morphology.distance_transform_edt(tep)
                    distmap[self.mask == 0] = np.finfo(np.float64).max
                    templist = np.nonzero(distmap == np.min(distmap))
                    templist = np.array(templist)
                    distrk = -1
                    distt = np.finfo(np.float64).max
                    for j in xrange(templist.shape[1]):
                        if distt > np.linalg.norm(np.array(templist)[:,j] - np.array(leftfromprev)[i,:]):
                            distt = np.linalg.norm(np.array(templist)[:,j] - np.array(leftfromprev)[i,:])
                            distrk = j

                    if distt > distmap.shape[0]/3.0:
                        tvalue = np.array(leftfromprev)[i,:]
                    else:
                        tvalue = np.array(templist)[:,distrk]

                    newCentersX = np.append(newCentersX, tvalue[0])
                    newCentersY = np.append(newCentersY, tvalue[1])

        #average the center position between from previous frame and current detection
        if len(parai) >=3:
            precentersX=parai[len(parai)-1][self.nComponents-2].params[1::6]
            precentersY=parai[len(parai)-1][self.nComponents-2].params[2::6]
            if len(precentersX)==2:
                order=[[0, 1], [1, 0]]
            else:
                order=[[0, 1, 2], [0, 2, 1],[1,2,0],[1,0,2],[2,1,0],[2,0,1]]
            temp=np.finfo(np.float64).max
            for m in xrange(len(order)):
                to=order[m]
                dist=0.0
                for n in xrange(len(to)):
                    dist+=np.linalg.norm(np.array([precentersX[n],precentersY[n]])-np.array([newCentersX[to[n]],newCentersY[to[n]]]))
                if dist < temp:
                    temp=dist
                    distrec=m
            newCentersX = np.round(0.5*newCentersX[order[distrec]]+ 0.5*precentersX)
            newCentersY = np.round(0.5*newCentersY[order[distrec]]+ 0.5*precentersY)

        for i in xrange(self.nComponents):
            if len(parai) < 3:# or (parai[len(parai) - 1][0].error - parai[len(parai) - 1][1].error) < 0:
                # self.params[6*i:6*i+6] = [max(self.th_weight_low, np.sqrt(self.image[newCentersX[i], newCentersY[i]])), newCentersX[i], newCentersY[i], self.th_sigma_high, self.th_sigma_high, 0.]
                self.params[6 * i:6 * i + 6] = [self.th_weight_low, newCentersX[i], newCentersY[i], self.th_sigma_high,
                                                self.th_sigma_high, 0.1]
            else:

                self.params[6 * i + 1:6 * i + 3] = [newCentersX[i], newCentersY[i]]
                #self.params[6*i+3:6*i+6] = [self.th_sigma_high,   self.th_sigma_high, 0.1]
                prevpa = [parai[len(parai) - 1][self.nComponents - 2].params[6 * i + 3:6 * i + 6],
                          parai[len(parai) - 2][self.nComponents - 2].params[6 * i + 3:6 * i + 6],
                          parai[len(parai) - 3][self.nComponents - 2].params[6 * i + 3:6 * i + 6]]
                k1 = np.array([[prevpa[0][0] ** 2, prevpa[0][0] * prevpa[0][1] * prevpa[0][2]],
                               [prevpa[0][0] * prevpa[0][1] * prevpa[0][2], prevpa[0][1] ** 2]])
                k2 = np.array([[prevpa[1][0] ** 2, prevpa[1][0] * prevpa[1][1] * prevpa[1][2]],
                               [prevpa[1][0] * prevpa[1][1] * prevpa[1][2], prevpa[1][1] ** 2]])
                k3 = np.array([[prevpa[2][0] ** 2, prevpa[2][0] * prevpa[2][1] * prevpa[2][2]],
                               [prevpa[2][0] * prevpa[2][1] * prevpa[2][2], prevpa[2][1] ** 2]])
                w1, v1 = np.linalg.eig(k1)
                idx = w1.argsort()
                w1 = w1[idx]
                v1 = v1[:, idx]
                w2, v2 = np.linalg.eig(k2)
                idx = w2.argsort()
                w2 = w2[idx]
                v2 = v2[:, idx]
                w3, v3 = np.linalg.eig(k3)
                idx = w3.argsort()
                w3 = w3[idx]
                v3 = v3[:, idx]

                predw = w1#0.6*w1 + 0.3*w2 + 0.1*w3
                predangle = np.arctan(v1[1][1]/v1[0][1])+(np.arctan(v1[1][1]/v1[0][1])-np.arctan(v2[1][1]/v2[0][1]))#0.6*np.arctan(v1[1][1]/v1[0][1]) + 0.3*np.arctan(v2[1][1]/v2[0][1]) + 0.1*np.arctan(v3[1][1]/v3[0][1])#
                predv = np.array([v1[0, :],[v1[0, 0]*np.tan(predangle+np.pi/2.), v1[0, 1]*np.tan(predangle)]])
                k = predv.dot(np.array([[predw[0],0],[0,predw[1]]])).dot(np.linalg.inv(predv))
                self.params[6*i+3:6*i+6] = [np.sqrt(k[0][0]), np.sqrt(k[1][1]), k[0][1]/np.sqrt(k[0][0])/np.sqrt(k[1][1])]

                self.params[6 * i] = self.th_weight_low
                #self.params[6 * i] = 1./(2*np.pi*self.params[6*i+3]*self.params[6*i+4]*np.sqrt(1-self.params[6*i+5]**2))

    def checkMoments(self):
        self.display_params()
        for i in xrange(self.nComponents):
            if self.params[6 * i] < self.th_weight_low:
                return False
            if self.mask[np.abs(self.params[6 * i + 1]), np.abs(self.params[6 * i + 2])] == 0:
                return False
            if np.abs(self.params[6 * i + 3]) > self.th_sigma_high or np.abs(
                    self.params[6 * i + 4]) > self.th_sigma_high:
                return False
        return True

    def getDistanceMap(self):
        new_mask = np.ones(self.mask.shape)
        new_mask[self.mask == 1] = 0
        map_ = scipy.ndimage.morphology.distance_transform_edt(new_mask)
        return map_

    def errorfunction(self, p, dx, dy):

        f = self.createGaussian(p)(dx, dy)
        err = abs(f - self.image)

        err[self.image == 0] = 0

        termG = np.sum((err) ** 2)
        if self.mask == None:
            termCenters = 0
        else:
            cost = 0
            for i in xrange(self.nComponents):
                xtmp, ytmp = int(p[6 * i + 1]), int(p[6 * i + 2])

                if np.abs(xtmp) >= self.shape[0] or np.abs(ytmp) >= self.shape[1]:
                    cost = np.sum(self.distanceMap)
                    print 'cost', xtmp, ytmp
                    break
                else:
                    cost += self.distanceMap[xtmp, ytmp]
            termCenters = np.sum(cost) / np.max(self.distanceMap)
        temp1 = np.sum((np.array(p[3::6]) - self.th_sigma_high) ** 2)/ self.nComponents#np.sum((p[3::6] - self.th_sigma_high) ** 2 / self.th_sigma_high ** 2) / self.nComponents
        temp2 = np.sum((np.array(p[4::6]) - self.th_sigma_high) ** 2 )/ self.nComponents#(np.sum((p[4::6] - self.th_sigma_high) ** 2 / self.th_sigma_high ** 2)) / self.nComponents
        termWeight = np.sum((np.array(p[0::6]) ** 2 - self.th_weight_low ** 2) ** 2)/ self.nComponents#(np.sum((p[0::6] ** 2 - self.th_weight_low ** 2) ** 2 / self.th_weight_low ** 2 ** 2)) / self.nComponents
        termSigma = (temp1 + temp2) / 2.0

        term = termG * ( 1 + termCenters + termSigma + termWeight )
        return term

    def fitgaussian(self):
        dx, dy = np.indices(self.shape)
        self.inipara=np.copy(self.params)
        out = fmin_powell(self.errorfunction, self.params, args=(dx, dy), full_output=True, maxiter=10000, maxfun=50000,
                          disp=False)
        # self.updateMoments(out[0])
        self.updateMoments_new(out[0])
        self.error = out[1]
        # out = minimize(self.errorfunction, self.params, args=(dx, dy), method='L-BFGS-B', options= {'disp': False, 'maxiter': 10000})
        # self.updateMoments_new(out['x'])
        # self.error = out['fun']


    def plot_gaussians3D(self, save=False, titlehist='', pathfig='', newfig=True):

        ax = extract.hist2d(titlehist, newfig=newfig)
        dx, dy = np.indices(self.shape)
        for n in xrange(0, len(self.params), 6):
            gaussunitaire = GaussianForFit(self.image, 1, params=self.params[n:n + 6])
            ax.scatter(gaussunitaire.params[1], gaussunitaire.params[2],
                       self.image[gaussunitaire.params[1], gaussunitaire.params[2]], color=self.colors[n % 5],
                       label="{0:.3f}".format(gaussunitaire.params[0]), alpha=0.7)
            ax.contour(dx, dy, gaussunitaire.gaussian, colors=self.colors[n % 5])
        if save:
            pylab.savefig(pathfig)

    def plot_signal3D(self, save=False, pathfig='', titlehist='', newfig=True):
        extract = utilities.ExtractImage(self.image)
        ax = extract.hist2d(titlehist, newfig=newfig)
        dx, dy = np.indices((self.size, self.size))
        f = self.createGaussian(self.params)
        ax.plot_wireframe(dx, dy, f(dx, dy), alpha=0.3, colors='g')
        if save:
            pylab.savefig(pathfig)

    def draw2D(self, title, image=[]):
        pylab.figure()
        if image == []:
            pylab.imshow(self.image, 'gray')
        else:
            pylab.imshow(image, 'gray')
        pylab.axis('off')
        pylab.autoscale(False)
        for i in xrange(self.nComponents):
            xeq = lambda t: self.params[6 * i + 3] * np.cos(t) * np.cos(self.params[6 * i + 5]) + self.params[
                                                                                                      6 * i + 4] * np.sin(
                t) * np.sin(self.params[6 * i + 5]) + self.params[6 * i + 1]
            yeq = lambda t: - self.params[6 * i + 3] * np.cos(t) * np.sin(self.params[6 * i + 5]) + self.params[
                                                                                                        6 * i + 4] * np.sin(
                t) * np.cos(self.params[6 * i + 5]) + self.params[6 * i + 2]
            t = np.linspace(0, 2 * np.pi, 100)
            x = xeq(t)
            y = yeq(t)
            pylab.scatter(self.params[6 * i + 2], self.params[6 * i + 1], color='k')
            pylab.plot(y.astype(int), x.astype(int), self.colors[i] + '-')
        pylab.savefig(title)
        pylab.close()

    def draw2D_new(self, title, image=[]):
        tt=np.array(self.image.shape)/50.
        pylab.figure(figsize=tt, dpi=50.)
        if image == []:
            pylab.imshow(self.image, 'gray')
        else:
            pylab.imshow(image, 'gray')
        pylab.axis('off')
        pylab.autoscale(False)
        for i in xrange(self.nComponents):

            k1 = np.array([[self.params[6 * i + 3] ** 2, self.params[6 * i + 3] * self.params[6 * i + 4] * self.params[6 * i + 5]],
                           [self.params[6 * i + 3] * self.params[6 * i + 4] * self.params[6 * i + 5], self.params[6 * i + 4] ** 2]])
            w1, v1 = np.linalg.eig(k1)
            idx = w1.argsort()
            w1 = w1[idx]
            v1 = v1[:, idx]
            angle=-(np.arctan(v1[1][1]/v1[0][1]))+np.pi#x+2*(pi/4-x)+pi/2#since in the image X and Y are inverted, so need to minus 90 degree and flip around pi/4
            #print angle
            w2 = np.zeros((1 , 2))
            w2[0,1] = np.sqrt(2)*np.max([self.params[6 * i + 3], self.params[6 * i + 4]])
            w2[0,0] = w2[0,1]*w1[0]/w1[1]

            xeq = lambda t: w2[0,1] * np.cos(t) * np.cos(angle) + w2[0,0] * np.sin(
                t) * np.sin(angle) + self.params[6 * i + 1]
            yeq = lambda t: - w2[0,1] * np.cos(t) * np.sin(angle) + w2[0,0] * np.sin(
                t) * np.cos(angle) + self.params[6 * i + 2]
            t = np.linspace(0, 2 * np.pi, 100)
            x = xeq(t)
            y = yeq(t)
            pylab.scatter(self.params[6 * i + 2], self.params[6 * i +1], color=self.colors[i], s=2)
            pylab.plot(y.astype(int), x.astype(int), self.colors[i] + '-')

            pylab.plot([self.params[6 * i + 2], yeq(0)],[self.params[6 * i + 1],xeq(0)],self.colors[i]+'-')
            #draw for inipara before optimization
            k1 = np.array([[self.inipara[6 * i + 3] ** 2, self.inipara[6 * i + 3] * self.inipara[6 * i + 4] * self.inipara[6 * i + 5]],
                           [self.inipara[6 * i + 3] * self.inipara[6 * i + 4] * self.inipara[6 * i + 5], self.inipara[6 * i + 4] ** 2]])
            w1, v1 = np.linalg.eig(k1)
            idx = w1.argsort()
            w1 = w1[idx]
            v1 = v1[:, idx]
            angle=-(np.arctan(v1[1][1]/v1[0][1]))+np.pi#x+2*(pi/4-x)+pi/2#since in the image X and Y are inverted, so need to minus 90 degree and flip around pi/4

            w2 = np.zeros((1 , 2))
            w2[0,1] = np.sqrt(2)*np.max([self.inipara[6 * i + 3], self.inipara[6 * i + 4]])
            w2[0,0] = w2[0,1]*w1[0]/w1[1]

            xeq = lambda t: w2[0,1] * np.cos(t) * np.cos(angle) + w2[0,0] * np.sin(
                t) * np.sin(angle) + self.inipara[6 * i + 1]
            yeq = lambda t: - w2[0,1] * np.cos(t) * np.sin(angle) + w2[0,0] * np.sin(
                t) * np.cos(angle) + self.inipara[6 * i + 2]
            t = np.linspace(0, 2 * np.pi, 100)
            x = xeq(t)
            y = yeq(t)
            x=x[0::5]
            y=y[0::5]
            pylab.scatter(self.inipara[6 * i + 2], self.inipara[6 * i +1], color=self.colors[i], s=1)
            pylab.plot(y.astype(int), x.astype(int), self.colors[i] + '--')

            pylab.plot([self.inipara[6 * i + 2], yeq(0)],[self.inipara[6 * i + 1],xeq(0)],self.colors[i]+'-')

        if self.nComponents==3:
            minind=np.argmin([np.linalg.norm(self.params[1:3]-self.params[7:9]),np.linalg.norm(self.params[1:3]-self.params[13:15]),np.linalg.norm(self.params[7:9]-self.params[13:15])])
            if minind==0:
                cenind=[1,7]
            elif minind==1:
                cenind=[1,13]
            else:
                cenind=[7,13]
            pylab.plot([self.params[cenind[0]+1],self.params[cenind[1]+1]], [self.params[cenind[0]],self.params[cenind[1]]], '-m')
        pylab.savefig(title,bbox_inches='tight', pad_inches=0)
        pylab.close()

    def minvalues(self, params2):
        postemp = [[1,2,7,8],[1,2,13,14],[7,8,13,14]]
        #minimum variance of intensity values between centers
        varinten3=[]
        for m in xrange(3):
            post=postemp[m]
            para3=self.params[post]
            if para3[0] > para3[2]:
                para3=[para3[2],para3[3],para3[0],para3[1]]
            para3=np.round(para3)
            inten=[]
            if para3[0]==para3[2]:
                x=para3[0]
                t1=np.min([int(para3[1]), int(para3[3])])
                t2=np.max([int(para3[1]), int(para3[3])])
                for y in xrange(t1, t2+1):
                    inten.append(self.image[np.round(x)][np.round(y)])
            else:
                for x in xrange(int(para3[0]), int(para3[2]+1)):
                    y = (para3[3]-para3[1])*(x-para3[0])/(para3[2]-para3[0])+para3[1]
                    if 0 <= np.round(y) < self.image.shape[1] and 0 <= np.round(x) < self.image.shape[0]:
                        inten.append(self.image[np.round(x)][np.round(y)])
            varinten3.append(np.var(inten))
        self.varmin = np.min(varinten3)
        # distance of 2 and 3 for centers and params: using mean
        center2_3=[]
        para2_3=[]
        for m in xrange(2):
            temp_center=[]
            temp_para=[]
            center2=params2[1+m*6:3+m*6]
            para2=params2[m*6:6+m*6]
            a=mat(para2)
            min_cdis3=np.mean([np.linalg.norm(center2-self.params[1:3]),np.linalg.norm(center2-self.params[7:9]),np.linalg.norm(center2-self.params[13:15])])
            center2_3.append(min_cdis3)
            trc=[]
            for n in xrange(3):
                para3=self.params[n*6:6+n*6]
                b=mat(para3)
                cossim = dot(a,b.T)/linalg.norm(a)/linalg.norm(b)
                cosdis = 1.-np.abs(cossim)
                trc.append(cosdis)
            para2_3.append(np.mean(trc))
        self.center23=np.mean(center2_3)
        self.param23=np.mean(para2_3)