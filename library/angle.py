import pylab, matplotlib
from random import randint
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import *
import scipy
import heapq
import os
from time import time
import math
from math import acos
from glob import glob
import subprocess

import utilities
import segmentationClass
import sequenceExtract
import GaussClasses

class AngleFinder(GaussClasses.GaussianObject):
    def __init__(self, nComponents, params):
        GaussClasses.GaussianObject.__init__(self, nComponents, params = params)
        if nComponents != 3:
            raise ValueError("Need a gaussianFitting Object with 3 components.")
        
    @staticmethod
    def euclidian_dist(x1, y1, x2, y2):
        return np.sqrt( (x2-x1)**2 + (y2-y1)**2 )
    @staticmethod
    def rotation(x, y, theta):
        x -= 50
        y -= 50
        theta = -theta  
        xprime = x * np.cos(theta) - y * np.sin(theta)
        yprime = y * np.sin(theta) + y * np.cos(theta)
        return xprime, yprime
        
    def getRotation(self, plot = True):
        if self.nComponents != 3:
            raise Exception("ERROR : not 3 centers")
        ## choice only surface based criteria
        areas = np.zeros(self.nComponents)
        for i in xrange(self.nComponents):
            areas[i] = self.params[6*i+3]*self.params[6*i+4]*np.pi
        goodcenters = range(self.nComponents)
        other = np.argmax(areas)
        goodcenters.pop(other)
        x3 = self.params[6*other+1]
        y3 = self.params[6*other+2]
        x1 = self.params[6*goodcenters[0]+1]
        y1 = self.params[6*goodcenters[0]+2]
        x2 = self.params[6*goodcenters[1]+1]
        y2 = self.params[6*goodcenters[1]+2]
        cx = (x1 + x2)/2.
        cy = (y1 + y2)/2.
        rot = acos((cy-y3) /1./ self.euclidian_dist(x3, y3, cx, cy))
        if x3-cx < 0:
            rot = -1.*rot
        if plot:
            pylab.subplot(2,2,1)
            pylab.plot([y3, cy], [x3, cx], 'm-', label = (x3-cx)>0)
            pylab.plot(y3, x3, 'go', label = (y3-cy)>0)
            pylab.axhline(y = x3, c= 'y', ls = '--')
            pylab.plot([y1, y2], [x1, x2], 'r+')
            pylab.subplot(2,2,2)
            pylab.plot(self.rotation(x3, y3, rot)[1], self.rotation(x3, y3, rot)[0], 'go')
            pylab.plot(self.rotation(cx, cy, rot)[1], self.rotation(cx, cy, rot)[0], 'ro')
            pylab.plot([self.rotation(x1, y1, rot)[1], self.rotation(x2, y2, rot)[1]], [self.rotation(x1, y1, rot)[0], self.rotation(x2, y2, rot)[0]], 'r+')
            pylab.xlim([-50., 50.])
            pylab.ylim([-50., 50.])
        return rot, [(x1,y1), (x2,y2), (x3,y3), (int((x3+cx)/2.), int((y3+cy)/2.))]
        
        
    def getAngle_gaussian(self, plot = True):
        if self.nComponents != 3:
            raise Exception("ERROR : not 3 centers")
        ## first only surface based criteria
        areas = np.zeros(self.nComponents)
        for i in xrange(self.nComponents):
            areas[i] = self.params[6*i+3]*self.params[6*i+4]*np.pi
        goodcenters = range(self.nComponents)
        other = np.argmax(areas)
        goodcenters.pop(other)
        x3 = self.params[6*other+1]
        y3 = self.params[6*other+2]
        x1 = self.params[6*goodcenters[0]+1]
        y1 = self.params[6*goodcenters[0]+2]
        x2 = self.params[6*goodcenters[1]+1]
        y2 = self.params[6*goodcenters[1]+2]
        cx = (x1 + x2)/2.
        cy = (y1 + y2)/2.
        d1 = self.euclidian_dist(cx, cy, x3, y3)
        d2 = self.euclidian_dist(cx, cy, x1, y1)
        dotproduct = (x3 - cx)*(self.params[6*goodcenters[0]+1] - cx) + \
        (y3 - cy)*(self.params[6*goodcenters[0]+2] - cy)
        a = acos(dotproduct/d1/d2)
        if a > np.pi/2.:
            a = np.pi - a
        if plot:
            self.Y = [y1,y2, y3, cy]
            self.X = [x1, x2,x3, cx]
            self.angle = a
        distDaughtercells = self.euclidian_dist(x1, y1, x2, y2)
        distMotherD1 = self.euclidian_dist(x1, y1, x3, y3)
        distMotherD2 = self.euclidian_dist(x2, y2, x3, y3)
        rightOrder = (distMotherD1 < distDaughtercells) & (distMotherD2 < distDaughtercells)
        return a, distDaughtercells, not rightOrder
        
    def draw(self):
        pylab.plot(self.Y[:2], self.X[:2], 'r-', label = "angle : {0:.0f}".format(self.angle*180/np.pi))
        pylab.plot(self.Y[2:], self.X[2:], 'y-')