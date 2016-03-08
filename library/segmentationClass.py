#!/usr/bin/env python

import cv2.cv as cv
import cv2
import numpy as np
import os
import pylab
from glob import glob
from random import randint
import scipy.ndimage as ndimage
from skimage.filter import threshold_otsu

import utilities
import GaussClasses
import sequenceExtract

from scipy import ndimage
from skimage.feature import peak_local_max
from skimage import img_as_float
from skimage.measure import label
from skimage.measure import regionprops
from skimage.segmentation import clear_border
from skimage.morphology import disk, erosion, dilation

class Segmentation(utilities.ExtractImage):
    def __init__(self, image, gauss=2, gaussMask=10, size=75, maxFilter=12):
        utilities.ExtractImage.__init__(self, image)
        self.gauss = gauss
        self.gaussMask = gaussMask
        self.extractSize = size
        self.maxFilter = maxFilter
        self.FG = None
        self.extract = None
        self.smooth = np.array(self.image)
        self.coorExtract = (None, None)
        self.centersX = []
        self.centersY = []
        self.nComponents = -1
        self.candi = np.array(self.image)
        self.areamax = 0
        self.areanum = 0
        self.viareanum = 0

    def defineFG(self, param):
        image = np.asarray(self.image, dtype = np.float)


        image= cv2.fastNlMeansDenoising(np.array(image, dtype = np.uint8),None,10,7,21)

        image = ndimage.filters.gaussian_filter(image, self.gaussMask)
        #ret, thresh = cv2.threshold(image, 1, 255 ,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        hist, indh = np.histogram(image, bins=10)
        temp=[i for i in xrange(len(hist)) if hist[i]>(image.size/2.0)]
        if len(temp) > 0 and temp[0]<5:
            if hist[temp[0]+1]>200:
                tempth = indh[temp[0]+1]
            else:
                tempth = indh[temp[0]+2]
            thresh = np.copy(image)
            thresh[thresh >= tempth] = 255
            thresh[thresh < tempth] =0
        else:
            ret, thresh = cv2.threshold(image, 1, 255 ,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


        mask=np.copy(thresh)
        mask[mask==0]=1
        mask[mask==255]=0

        if len(param):
            temp = np.copy(mask)
            label_img=label(temp)
            regions = regionprops(label_img)
            for props in regions:
                x0, y0 = props.centroid
                md = np.min([np.linalg.norm(np.array([param[1],param[2]])-np.array([x0,y0])),
                             np.linalg.norm(np.array([param[7],param[8]])-np.array([x0,y0])),
                             np.linalg.norm(np.array([param[13],param[14]])-np.array([x0,y0]))])
                if (y0 < 20 or x0 < 20 or (label_img.shape[0]-x0) < 20 or (label_img.shape[1]-y0) < 20) and md > mask.shape[0]/3:
                    label_img[label_img==props.label] = 0
            label_img[label_img>0] = 1
            mask=np.copy(label_img)
        else:
            temp = np.copy(mask)
            label_img=label(temp)
            regions = regionprops(label_img)
            while len(regions)>0 and regions[0].area > label_img.size/3.:
                label_img -= 1
                label_img[label_img<0] = 0
                regions = regionprops(label_img)

            for props in regions:
                x0, y0 = props.centroid
                if (y0 < 20 or x0 < 20 or (label_img.shape[0]-x0) < 20 or (label_img.shape[1]-y0) < 20):
                    label_img[label_img==props.label] = 0

            label_img[label_img>0] = 1

            mask=np.copy(label_img)
        self.FG = mask
        self.candi = image
        self.smooth= np.array(image)
        self.extract= np.array(image)
        #self.candi [self.FG == 0] = 0

        cleared=np.copy(mask)
        #clear_border(cleared)
        label_image = label(cleared)
        rept = regionprops(label_image)
        areamax=0
        viareanum=0
        for i in xrange(len(rept)):
            if rept[i].area > areamax:
                areamax = rept[i].area
            viareanum += np.max([np.floor(rept[i].area/14./7./np.pi),1])
        self.areamax = areamax
        self.areanum = len(rept)
        self.viareanum = viareanum
      
    def defineExtraction(self):
        ## original size, self.extractSize : wanted size of the extract
        size = len(self.FG[0]) 
        
        px = np.sum(self.FG, axis = 1)
        py = np.sum(self.FG, axis = 0)
        xmin = np.argmax((px>0))
        xmax = np.argmax((px[xmin+1:] ==0.)) + xmin
        ymin = np.argmax((py>0))
        ymax = np.argmax((py[ymin+1:]==0.)) + ymin
        xcenter = int((xmin+xmax)/2.)
        ycenter = int((ymin+ymax)/2.)

        if xcenter - self.extractSize/2 < 0:
            xmin = 0
        elif xcenter + self.extractSize/2 >= self.image.shape[0] :
            xmin = size - self.extractSize - 1
        else :
            xmin = xcenter - self.extractSize/2
        if ycenter - self.extractSize/2 < 0:
            ymin = 0
        elif ycenter + self.extractSize/2 >= self.image.shape[0] :
            ymin = size - self.extractSize - 1
        else :
            ymin = ycenter - self.extractSize/2

        self.coorExtract = (xmin, ymin)
        
        
    def smooth_corr(self):
        dst = ndimage.filters.gaussian_filter(self.image, self.gauss)
        self.defineExtraction()
        xmin, ymin = self.coorExtract
        bg = np.array(dst)
        val = 2 * np.max(dst)
        bg[xmin:xmin+self.extractSize, ymin:ymin+self.extractSize] = val
        bg = bg [ bg != val]
        smooth_CORRbg = dst - np.mean(bg)
        smooth_CORRbg [smooth_CORRbg < 0] = 0
        self.extract = smooth_CORRbg[xmin:xmin+self.extractSize, ymin:ymin+self.extractSize]
        self.smooth = np.array(smooth_CORRbg)
        
    
    def findMaximaOnFG(self, param):
        self.defineFG(param)
        #self.smooth_corr()
        self.coorExtract = [0, 0]
        xmin, ymin = self.coorExtract


        img=self.candi
        img [self.FG ==0] =0

        im = img_as_float(img)
        image_max = ndimage.maximum_filter(im, size=10, mode='constant')
        coordinates = peak_local_max(im, min_distance=10)

        tep=np.zeros(self.candi.shape)
        for i,ide in enumerate(coordinates):
            tep[ide[0],ide[1]] = self.candi[ide[0],ide[1]]
        lbl = ndimage.label(tep)[0]
        centerc = np.round(ndimage.measurements.center_of_mass(tep, lbl, range(1,np.max(lbl)+1)))
        if centerc.size > 0:
            self.centersX = centerc[:,0].astype(int)
            self.centersY = centerc[:,1].astype(int)
        self.nComponents = len(self.centersX)
