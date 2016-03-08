import matplotlib

import cv2
import cv2.cv as cv
import os
import numpy as np
import pylab
from glob import glob
from time import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import *

from skimage import exposure
from skimage.measure import label
from skimage.measure import regionprops
import segmentationClass
import GaussClasses
import utilities
import library.searchEventErrorfit as searchEventErrorfit
import library.angle as angle
from scipy import misc
import library.infofiles as infofiles

class Sequence():
    def __init__(self, filesSeq, fileout):
        self.filesSeq = filesSeq
        self.lengthSeq = len(filesSeq)
        self.listObj = []
        self.fileout = fileout

    def studySeq_testeachfeature(self, fct, title = '', **kwargs):
        divfr_b = -1
        ang = -1
        f1=[]
        f2=[]
        f3=[]
        fgeo=[]
        maxind=-1
        numstd=-1
        reall=[-1,-1,-1,-1,-1,-1,-1]


        im16bit_all=[]
        im8bit_all=[]
        # read all frames in video and normalize
        for index, imfile in enumerate(self.filesSeq):
            image = utilities.Extract(imfile).image
            im16bit_all.append(image)
        maxt=np.max(im16bit_all)
        mint=np.min(im16bit_all)
        for index, imfile in enumerate(self.filesSeq):
            im8bit=(im16bit_all[index]-mint)*1./(maxt-mint)*255.
            im8bit_all.append(im8bit)
        flagall=np.zeros(len(self.filesSeq))
        # Gaussian fitting to each frame
        for index, imfile in enumerate(self.filesSeq):

            self.flag = 0
            print "t = ", index
            self.fileout.write( "#### {0}\n".format(imfile.split('\\')[-1]))

            image = im8bit_all[index]
            im16bit = im16bit_all[index]

            if index == 0: # gaussian fitting
                fitObject = fct(image, title = "{0}_t{1:03d}".format(title, index), **kwargs)
            else:
                fitObject = fct(image, title = "{0}_t{1:03d}".format(title, index), **kwargs)

            flagall[index]=self.flag
            divfr = -1
            # #on-line identify the division frame time
            # if self.flag == 1:
            #     if index > 0:
            #         self.listObj.append(self.listObj[index-1])
            #         continue
            #     else:
            #         return divfr_b, ang, maxind, numstd, f1, f2, f3, fgeo

            # armsize=10
            # if index >= 5 and len(regionprops(self.listObj[-2][0].mask))<3:
            #     if divfr_b==-1:
            #         divfr, ang = self.identi(armsize)
            #         if divfr > -1:
            #             divfr_b=divfr

            self.listObj.append(fitObject)

        # off-line identification
        #divfr, ang,  maxind, numstd,f1, f2, f3, fgeo = self.identi_offline(flagall, **kwargs)
        #return divfr, ang, maxind, numstd, f1, f2, f3, fgeo

        reall = self.identi_offline_testeachfeature(flagall, **kwargs)
        return reall

    def studySeq(self, fct, title = '', **kwargs):
        divfr_b = -1
        ang = -1
        f1=[]
        f2=[]
        f3=[]
        fgeo=[]
        maxind=-1
        numstd=-1
        reall=[-1,-1,-1,-1,-1,-1,-1]


        im16bit_all=[]
        im8bit_all=[]
        # read all frames in video and normalize
        for index, imfile in enumerate(self.filesSeq):
            image = utilities.Extract(imfile).image
            im16bit_all.append(image)
        maxt=np.max(im16bit_all)
        mint=np.min(im16bit_all)
        for index, imfile in enumerate(self.filesSeq):
            im8bit=(im16bit_all[index]-mint)*1./(maxt-mint)*255.
            im8bit_all.append(im8bit)
        flagall=np.zeros(len(self.filesSeq))
        # Gaussian fitting to each frame
        for index, imfile in enumerate(self.filesSeq):

            self.flag = 0
            print "t = ", index
            self.fileout.write( "#### {0}\n".format(imfile.split('\\')[-1]))

            image = im8bit_all[index]
            im16bit = im16bit_all[index]

            if index == 0: # gaussian fitting
                fitObject = fct(image, title = "{0}_t{1:03d}".format(title, index), **kwargs)
            else:
                fitObject = fct(image, title = "{0}_t{1:03d}".format(title, index), **kwargs)

            flagall[index]=self.flag
            divfr = -1


            self.listObj.append(fitObject)

        # off-line identification
        divfr, ang,  maxind, numstd,f1, f2, f3, fgeo = self.identi_offline(flagall, **kwargs)
        return divfr, ang, maxind, numstd, f1, f2, f3, fgeo


    def identi_offline_testeachfeature(self,flagall, **kwargs): # off-line detection of cell division
        frnum = len(self.listObj)
        err2 = np.zeros(frnum)
        err3 = np.zeros(frnum)
        rou1 = np.zeros(frnum)
        rou2 = np.zeros(frnum)
        para3 = []
        varmin=[]
        for i in xrange(frnum):
            err2[i] = self.listObj[i][0].error
            err3[i] = self.listObj[i][1].error
            rou1[i] = self.listObj[i][0].params[5]
            rou2[i] = self.listObj[i][0].params[11]
            para3.append(self.listObj[i][1].params)
            varmin.append(self.listObj[i][1].varmin)
        diff = (err2-err3)/err3
        diff /= np.max(diff)
        diff = (diff - np.min(diff))/(np.max(diff)-np.min(diff))
        para3 = np.array(para3)
        #max and min distance of centers
        mincenter_3=[]
        for k in xrange(para3.shape[0]):
            min_cdis3=np.min([np.linalg.norm(para3[k,1:3]-para3[k,7:9]),np.linalg.norm(para3[k,1:3]-para3[k,13:15]),np.linalg.norm(para3[k,7:9]-para3[k,13:15])])
            mincenter_3.append(min_cdis3)
        #max and min distance of set of parameters
        para3_b=np.copy(para3)
        for m in xrange(6):
            max_t=np.max(para3[:,m::6])
            min_t=np.min(para3[:,m::6])
            para3_b[:,m::6]=(para3_b[:,m::6]-min_t)/(max_t-min_t)
        minpara_3=[]
        for k in xrange(para3_b.shape[0]):
            min_cdis3=np.min([np.linalg.norm(para3_b[k,0:6]-para3_b[k,6:12]),np.linalg.norm(para3_b[k,0:6]-para3_b[k,12:18]),np.linalg.norm(para3_b[k,6:12]-para3_b[k,12:18])])
            minpara_3.append(min_cdis3)
        armsize=0
        traj = searchEventErrorfit.Trajectory(diff, armsize, mincenter_3, minpara_3, varmin)
        reall = traj.decision_lyb_off_testeachfeature()  # use product of three features to identify
        reall2=[]

        for refr in reall:

            if flagall[refr] == 1:
                refr = -1

            refr2 = -1
            ang = -1
            if refr>-1: # check the distance of cell
                min_cdis2=np.min([np.linalg.norm(np.array([self.listObj[refr][1].params[1],self.listObj[refr][1].params[2]])-np.array([self.listObj[refr][1].params[7],self.listObj[refr][1].params[8]])),
                                    np.linalg.norm(np.array([self.listObj[refr][1].params[1],self.listObj[refr][1].params[2]])-np.array([self.listObj[refr][1].params[13],self.listObj[refr][1].params[14]])),
                                    np.linalg.norm(np.array([self.listObj[refr][1].params[7],self.listObj[refr][1].params[8]])-np.array([self.listObj[refr][1].params[13],self.listObj[refr][1].params[14]]))])
                dis2 = np.linalg.norm(np.array([self.listObj[refr][0].params[1],self.listObj[refr][0].params[2]])-np.array([self.listObj[refr][0].params[7],self.listObj[refr][0].params[8]]))

                mov = np.linalg.norm(((np.array([self.listObj[refr-1][1].params[1],self.listObj[refr-1][1].params[2]])+np.array([self.listObj[refr-1][1].params[7],self.listObj[refr-1][1].params[8]])+np.array([
                                        self.listObj[refr-1][1].params[13],self.listObj[refr-1][1].params[14]]))/3.0-(np.array([self.listObj[refr][1].params[1], self.listObj[refr][1].params[2]])+np.array([
                                        self.listObj[refr][1].params[7],self.listObj[refr][1].params[8]])+np.array([self.listObj[refr][1].params[13],self.listObj[refr][1].params[14]]))/3.0))
                tr = 0
                for i in xrange(3):
                    tr2 = self.listObj[refr][0].mask.shape[0]
                    for j in xrange(3):
                        temp=np.linalg.norm(np.array([self.listObj[refr-1][1].params[1+i*6],self.listObj[refr-1][1].params[2+i*6]])-np.array([self.listObj[refr][1].params[1+j*6],self.listObj[refr][1].params[2+j*6]]))
                        if tr2 > temp:
                            tr2 = temp
                    if tr2>tr and tr2 > mov/5. and tr2 < mov*5.:
                        tr = tr2

                if min_cdis2 > (float(kwargs['maxFilter'])+3.)*0.75 and dis2>(float(kwargs['maxFilter'])+3.):
                    refr2 = refr

            reall2.append(refr2)
        return reall2

    def identi_offline(self,flagall, **kwargs): # off-line detection of cell division
        frnum = len(self.listObj)
        err2 = np.zeros(frnum)
        err3 = np.zeros(frnum)
        rou1 = np.zeros(frnum)
        rou2 = np.zeros(frnum)
        para3 = []
        varmin=[]
        for i in xrange(frnum):
            err2[i] = self.listObj[i][0].error
            err3[i] = self.listObj[i][1].error
            rou1[i] = self.listObj[i][0].params[5]
            rou2[i] = self.listObj[i][0].params[11]
            para3.append(self.listObj[i][1].params)
            varmin.append(self.listObj[i][1].varmin)
        diff = (err2-err3)/err3
        diff /= np.max(diff)
        diff = (diff - np.min(diff))/(np.max(diff)-np.min(diff))
        para3 = np.array(para3)
        #max and min distance of centers
        mincenter_3=[]
        for k in xrange(para3.shape[0]):
            min_cdis3=np.min([np.linalg.norm(para3[k,1:3]-para3[k,7:9]),np.linalg.norm(para3[k,1:3]-para3[k,13:15]),np.linalg.norm(para3[k,7:9]-para3[k,13:15])])
            mincenter_3.append(min_cdis3)
        #max and min distance of set of parameters
        para3_b=np.copy(para3)
        for m in xrange(6):
            max_t=np.max(para3[:,m::6])
            min_t=np.min(para3[:,m::6])
            para3_b[:,m::6]=(para3_b[:,m::6]-min_t)/(max_t-min_t)
        minpara_3=[]
        for k in xrange(para3_b.shape[0]):
            min_cdis3=np.min([np.linalg.norm(para3_b[k,0:6]-para3_b[k,6:12]),np.linalg.norm(para3_b[k,0:6]-para3_b[k,12:18]),np.linalg.norm(para3_b[k,6:12]-para3_b[k,12:18])])
            minpara_3.append(min_cdis3)
        armsize=0
        traj = searchEventErrorfit.Trajectory(diff, armsize, mincenter_3, minpara_3, varmin)
        refr, maxind, numstd = traj.decision_lyb_off()  # use product of three features to identify
        if flagall[refr] == 1:
            refr = -1

        refr2 = -1
        ang = -1
        if refr>-1: # check the distance of cell
            min_cdis2=np.min([np.linalg.norm(np.array([self.listObj[refr][1].params[1],self.listObj[refr][1].params[2]])-np.array([self.listObj[refr][1].params[7],self.listObj[refr][1].params[8]])),
                                np.linalg.norm(np.array([self.listObj[refr][1].params[1],self.listObj[refr][1].params[2]])-np.array([self.listObj[refr][1].params[13],self.listObj[refr][1].params[14]])),
                                np.linalg.norm(np.array([self.listObj[refr][1].params[7],self.listObj[refr][1].params[8]])-np.array([self.listObj[refr][1].params[13],self.listObj[refr][1].params[14]]))])
            dis2 = np.linalg.norm(np.array([self.listObj[refr][0].params[1],self.listObj[refr][0].params[2]])-np.array([self.listObj[refr][0].params[7],self.listObj[refr][0].params[8]]))

            mov = np.linalg.norm(((np.array([self.listObj[refr-1][1].params[1],self.listObj[refr-1][1].params[2]])+np.array([self.listObj[refr-1][1].params[7],self.listObj[refr-1][1].params[8]])+np.array([
                                    self.listObj[refr-1][1].params[13],self.listObj[refr-1][1].params[14]]))/3.0-(np.array([self.listObj[refr][1].params[1], self.listObj[refr][1].params[2]])+np.array([
                                    self.listObj[refr][1].params[7],self.listObj[refr][1].params[8]])+np.array([self.listObj[refr][1].params[13],self.listObj[refr][1].params[14]]))/3.0))
            tr = 0
            for i in xrange(3):
                tr2 = self.listObj[refr][0].mask.shape[0]
                for j in xrange(3):
                    temp=np.linalg.norm(np.array([self.listObj[refr-1][1].params[1+i*6],self.listObj[refr-1][1].params[2+i*6]])-np.array([self.listObj[refr][1].params[1+j*6],self.listObj[refr][1].params[2+j*6]]))
                    if tr2 > temp:
                        tr2 = temp
                if tr2>tr and tr2 > mov/5. and tr2 < mov*5.:
                    tr = tr2

            if min_cdis2 > (float(kwargs['maxFilter'])+3.)*0.75 and dis2>(float(kwargs['maxFilter'])+3.):
                refr2 = refr

                ang = self.divideangle(refr2)
        return refr2, ang, maxind, numstd, traj.f1, traj.f2, traj.f3, traj.fgeo

    def identi(self, armsize): # on-line detection of cell division
        frnum = len(self.listObj)
        err2 = np.zeros(frnum)
        err3 = np.zeros(frnum)
        rou1 = np.zeros(frnum)
        rou2 = np.zeros(frnum)
        para3 = []
        varmin=[]
        for i in xrange(frnum):
            err2[i] = self.listObj[i][0].error
            err3[i] = self.listObj[i][1].error
            rou1[i] = self.listObj[i][0].params[5]
            rou2[i] = self.listObj[i][0].params[11]
            para3.append(self.listObj[i][1].params)
            varmin.append(self.listObj[i][1].varmin)
        diff = (err2-err3)/err3
        diff /= np.max(diff)
        diff = (diff - np.min(diff))/(np.max(diff)-np.min(diff))
        para3 = np.array(para3)
        #if np.min(diff) < 0:
        #    diff += np.min(diff)*-1.
        #    diff /= np.max(diff)
        #max and min distance of centers
        mincenter_3=[]
        for k in xrange(para3.shape[0]):
            min_cdis3=np.min([np.linalg.norm(para3[k,1:3]-para3[k,7:9]),np.linalg.norm(para3[k,1:3]-para3[k,13:15]),np.linalg.norm(para3[k,7:9]-para3[k,13:15])])
            mincenter_3.append(min_cdis3)

        #max and min distance of set of parameters
        para3_b=np.copy(para3)
        for m in xrange(6):
            max_t=np.max(para3[:,m::6])
            min_t=np.min(para3[:,m::6])
            para3_b[:,m::6]=(para3_b[:,m::6]-min_t)/(max_t-min_t)
        minpara_3=[]
        for k in xrange(para3_b.shape[0]):
            min_cdis3=np.min([np.linalg.norm(para3_b[k,0:6]-para3_b[k,6:12]),np.linalg.norm(para3_b[k,0:6]-para3_b[k,12:18]),np.linalg.norm(para3_b[k,6:12]-para3_b[k,12:18])])
            minpara_3.append(min_cdis3)

        if np.max(rou1)/np.percentile(rou1, 95) > np.max(rou2)/np.percentile(rou2, 95):
            ro = (rou1-np.min(rou1))/(np.max(rou1)-np.min(rou1))
        else:
            ro = (rou2-np.min(rou2))/(np.max(rou2)-np.min(rou2))

        traj = searchEventErrorfit.Trajectory(diff, armsize, mincenter_3, minpara_3, varmin)
        #peak of difference of error2 and error3
        refr = traj.decision_lyb5()  # use geometric mean to identify

        refr2 = -1
        ang = -1

        if refr != -1 and (len(self.listObj)>0 and len(regionprops(label(self.listObj[refr-1][0].mask)))<=3):
            min_cdis2=np.min([np.linalg.norm(np.array([self.listObj[refr][1].params[1],self.listObj[refr][1].params[2]])-np.array([self.listObj[refr][1].params[7],self.listObj[refr][1].params[8]])),
                                np.linalg.norm(np.array([self.listObj[refr][1].params[1],self.listObj[refr][1].params[2]])-np.array([self.listObj[refr][1].params[13],self.listObj[refr][1].params[14]])),
                                np.linalg.norm(np.array([self.listObj[refr][1].params[7],self.listObj[refr][1].params[8]])-np.array([self.listObj[refr][1].params[13],self.listObj[refr][1].params[14]]))])
            max_cdis2=np.max([np.linalg.norm(np.array([self.listObj[refr][1].params[1],self.listObj[refr][1].params[2]])-np.array([self.listObj[refr][1].params[7],self.listObj[refr][1].params[8]])),
                                np.linalg.norm(np.array([self.listObj[refr][1].params[1],self.listObj[refr][1].params[2]])-np.array([self.listObj[refr][1].params[13],self.listObj[refr][1].params[14]])),
                                np.linalg.norm(np.array([self.listObj[refr][1].params[7],self.listObj[refr][1].params[8]])-np.array([self.listObj[refr][1].params[13],self.listObj[refr][1].params[14]]))])
            dis2 = np.linalg.norm(np.array([self.listObj[refr][0].params[1],self.listObj[refr][0].params[2]])-np.array([self.listObj[refr][0].params[7],self.listObj[refr][0].params[8]]))

            max_cdis3=np.max([np.linalg.norm(np.array([self.listObj[refr-1][1].params[1],self.listObj[refr-1][1].params[2]])-np.array([self.listObj[refr-1][1].params[7],self.listObj[refr-1][1].params[8]])),
                                np.linalg.norm(np.array([self.listObj[refr-1][1].params[1],self.listObj[refr-1][1].params[2]])-np.array([self.listObj[refr-1][1].params[13],self.listObj[refr-1][1].params[14]])),
                                np.linalg.norm(np.array([self.listObj[refr-1][1].params[7],self.listObj[refr-1][1].params[8]])-np.array([self.listObj[refr-1][1].params[13],self.listObj[refr-1][1].params[14]]))])
            mov = np.linalg.norm(((np.array([self.listObj[refr-1][1].params[1],self.listObj[refr-1][1].params[2]])+np.array([self.listObj[refr-1][1].params[7],self.listObj[refr-1][1].params[8]])+np.array([
                                    self.listObj[refr-1][1].params[13],self.listObj[refr-1][1].params[14]]))/3.0-(np.array([self.listObj[refr][1].params[1], self.listObj[refr][1].params[2]])+np.array([
                                    self.listObj[refr][1].params[7],self.listObj[refr][1].params[8]])+np.array([self.listObj[refr][1].params[13],self.listObj[refr][1].params[14]]))/3.0))
            tr = 0
            for i in xrange(3):
                tr2 = self.listObj[refr][0].mask.shape[0]
                for j in xrange(3):
                    temp=np.linalg.norm(np.array([self.listObj[refr-1][1].params[1+i*6],self.listObj[refr-1][1].params[2+i*6]])-np.array([self.listObj[refr][1].params[1+j*6],self.listObj[refr][1].params[2+j*6]]))
                    if tr2 > temp:
                        tr2 = temp
                if tr2>tr and tr2 > mov/5. and tr2 < mov*5.:
                    tr = tr2

            if min_cdis2 > (12.+3.)*0.75 and dis2>(12.+3.):
                #peak of rou of 2 components
                etemp_p = np.argmax(diff[refr-2:refr+3]*0.6+ro[refr-2:refr+3]*0.4)#(etemp[refr-2:refr+3])
                refr2 = etemp_p+(refr-2)
                min_cdis=np.linalg.norm(np.array([self.listObj[refr2][0].params[1],self.listObj[refr2][0].params[2]])-np.array([self.listObj[refr2][0].params[7],self.listObj[refr2][0].params[8]]))
                if min_cdis < 10:
                    refr2 = refr

                ang = self.divideangle(refr2)

        return refr2, ang




    def divideangle(self, tSelect): # compute division angle after detecting cell division

        if tSelect > 0 and tSelect < len(self.listObj):
            params2 = self.listObj[tSelect][0].params
            params3 = self.listObj[tSelect][1].params
            angleFind = angle.AngleFinder(3, params3)
            a, distDaughtercells, rightOrder = angleFind.getAngle_gaussian()

            th_dist=25
            if rightOrder and (distDaughtercells <= th_dist):

                #fileout.write(",TAU{0},ORDER{1},DIST{2},T{3},A{4}\n".format(traj.tau, str(rightOrder)[0], distDaughtercells, tSelect, a))
                plotflag=1
                if plotflag:
                    pylab.figure()
                    for indexIm in range(-2, 1):
                        pylab.subplot(2, 3, indexIm+3)
                        if tSelect+indexIm > 0:

                            imfilePrevious = glob("edga_*_t{0:04d}.tif".format(tSelect+indexIm))
                            try:
                                imagePrevious = cv2.imread(imfilePrevious[0], cv.CV_LOAD_IMAGE_UNCHANGED)
                                if len(imagePrevious.shape) == 3:
                                    print imagePrevious.shape, len(imagePrevious.shape)
                                    imagePrevious = imagePrevious[:,:,0]
                                pylab.imshow(imagePrevious, 'gray')
                                pylab.axis('off')
                            except IndexError:
                                print "no image at ", "edga_*s{0}_t{1:04d}.tif".format( 'dir', tSelect+indexIm)
                                pass
                            pylab.title("t = {0}".format(tSelect+indexIm))
                    imfile = glob("edga_*_t{0:04d}.tif".format(tSelect+indexIm))
                    if len(imfile) > 1:
                        print "too many images"
                        raise IndexError
                    try:
                        image = cv2.imread(imfile[0], cv.CV_LOAD_IMAGE_UNCHANGED)
                        if len(image.shape) == 3:
                            image = image[:,:,0]
                        pylab.subplot(2,3,4)
                        pylab.imshow(image, 'gray')
                        pylab.axis('off')
                        pylab.autoscale(False)
                        angleFind.draw()
                        pylab.title("angle={0:.0f} degree".format(a*180./np.pi))
                        pylab.subplot(2,3,5)
                        pylab.imshow(image, 'gray')
                        pylab.axis('off')
                        pylab.autoscale(False)
                        angleFind.draw2D_new()
                        pylab.subplot(2,3,6)
                        pylab.imshow(image, 'gray')
                        pylab.axis('off')
                        pylab.autoscale(False)
                        GaussClasses.GaussianObject(2, params=params2).draw2D_new()
                    except IndexError:
                        print "no image at : ", "edga_*s{0}_t{1:04d}.tif".format('dir', tSelect+indexIm)
                    pylab.title("t = {0}".format(tSelect))
                    #pylab.legend()
                    #pylab.savefig("../angle_s{0}_x{1:03d}_y{2:03d}_t{3:03d}.png".format('dir', int(x), int(y), int(tSelect)))
                    pylab.savefig("../angle_t{0:03d}.png".format(int(tSelect)))
                    pylab.close()
            return a*180./np.pi
            #else:
            #    fileout.write(",TAU{0},ORDER{1},DIST{2},T{3}\n".format(traj.tau, str(rightOrder)[0], distDaughtercells, tSelect))


    def treatImage(self, image, paramsFitting = None, gauss = 2, gaussMask = 4, size = 75, maxFilter = 12, th_sigma_high = 12., th_weight_low = 1000., plot = False, title = ''):
        fitObject = []
        seg = segmentationClass.Segmentation(image, gauss = gauss, gaussMask = gaussMask, size = size, maxFilter = maxFilter)
        '''
        # save the smoothed images
        seg.findMaximaOnFG()
        smfolder='/'.join(title.split('/')[0:-2])+'/smoothedimage'
        if not os.path.exists(smfolder):
            os.makedirs(smfolder)
        smfile =smfolder + '/' + title.split('_')[-1] + '.tif'
        cv2.imwrite(smfile, seg.smooth)
        '''
        if paramsFitting == None:
            if len(self.listObj)>0:
                seg.findMaximaOnFG(self.listObj[len(self.listObj)-1][1].params)
            else:
                seg.findMaximaOnFG([])


            if seg.areamax > (maxFilter+2.) **2*np.pi*2 or np.sum(seg.FG)>(maxFilter+2.)**2*np.pi*2.5 or seg.areanum > 3 or seg.areanum==0 or seg.viareanum>3 or len(seg.centersX)>3: # check if there are many cell in the frame in advance
                self.flag = 1
                #return

            for nComponents in range(2,4):
                #print "nComponents", nComponents
                gaussObject = GaussClasses.GaussianForFit(seg.extract, nComponents, mask = seg.FG, params = None, th_sigma_high = th_sigma_high, th_weight_low = th_weight_low)
                #no detection of center.the center is from previous results

                gaussObject.initMoments(seg.centersX, seg.centersY, self.listObj) # initilize the center for guassian component
                gaussObject.fitgaussian()# gaussian fitting -- main code of gaussian fitting
                gaussObject.changeCenters(*seg.coorExtract)


                if nComponents==3:
                    gaussObject.minvalues(fitObject[0].params)
                if plot:
                    gaussObject.draw2D_new("{0}_nComp{1}.png".format(title, nComponents), image = seg.image)
                    misc.imsave("{0}_mask.png".format(title),seg.FG)
                self.fileout.write("{0},".format(gaussObject.nComponents))
                self.fileout.write(",".join([str(p) for p in gaussObject.params]))
                self.fileout.write(",{0}\n".format(gaussObject.error))
                fitObject.append(gaussObject)


        else:
            seg.findMaximaOnFG()
            for indi, oldGaussObj in enumerate(paramsFitting):
                #print "nComponents", oldGaussObj.nComponents
                gaussObject = GaussClasses.GaussianForFit(seg.extract, oldGaussObj.nComponents, mask = seg.FG, params = oldGaussObj.params, th_sigma_high = th_sigma_high, th_weight_low = th_weight_low)
                gaussObject.changeCenters(- seg.coorExtract[0], - seg.coorExtract[1])
                gaussObject.fitgaussian()
                gaussObject_new = GaussClasses.GaussianForFit(seg.extract, oldGaussObj.nComponents, mask = seg.FG, params = None, th_sigma_high = th_sigma_high, th_weight_low = th_weight_low)
                gaussObject_new.initMoments(seg.centersX, seg.centersY)
                gaussObject_new.fitgaussian()
                if gaussObject_new.error < gaussObject.error and gaussObject_new.checkMoments() :
                    gaussObject_Selected = gaussObject_new
                else:
                    gaussObject_Selected = gaussObject
                gaussObject_Selected.changeCenters(*seg.coorExtract)
                if plot:
                    gaussObject_Selected.draw2D("{0}_nComp{1}.png".format(title, gaussObject.nComponents), image = seg.smooth)
                self.fileout.write("{0},".format(gaussObject_Selected.nComponents))
                self.fileout.write(",".join([str(p) for p in gaussObject_Selected.params]))
                self.fileout.write(",{0}\n".format(gaussObject_Selected.error))
                fitObject.append(gaussObject_Selected)
        return fitObject

    def plotParams(self, th_sigma_high, th_weight_low, size, title = ''):
        colors = ['r', 'y', 'g', 'c', 'm', 'b']
        paramNames = ['Weight', 'CenterX', 'CenterY', 'SigmaX', 'SigmaY']
        low_lims = [th_weight_low, 0, 0, 0,0]
        high_lims = [None, size, size, th_sigma_high, th_sigma_high]
        fig1 = pylab.figure(figsize = (15,15))
        parameters2, parameters3 = [], []
        for i in xrange(self.lengthSeq):
            for obj in self.listObj[i]:
                if obj.nComponents == 2:
                    parameters2.extend(obj.params)
                elif obj.nComponents == 3:
                    parameters3.extend(obj.params)
        for nParam in range(5):
            pylab.subplot(3,2,nParam+1)
            pylab.plot(range(self.lengthSeq), np.abs(parameters2[nParam::12]), 'b-')
            pylab.plot(range(self.lengthSeq), np.abs(parameters2[6+nParam::12]), 'm-')
            pylab.axhline(y = low_lims[nParam], color = 'r', ls = '-', lw = 5)
            if high_lims[nParam] != 0:
                pylab.axhline(y = high_lims[nParam], color = 'r', ls = '-', lw = 5)
            pylab.title(paramNames[nParam])
        pylab.savefig("{0}_params2.png".format(title))
        pylab.close()
        fig1 = pylab.figure(figsize = (20,15))
        for nParam in range(5):
            pylab.subplot(3,2,nParam+1)
            pylab.plot(range(self.lengthSeq), np.abs(parameters3[nParam::18]), 'b-')
            pylab.plot(range(self.lengthSeq), np.abs(parameters3[6+nParam::18]), 'm-')
            pylab.plot(range(self.lengthSeq), np.abs(parameters3[12+nParam::18]), 'g-')
            pylab.axhline(y = low_lims[nParam], color = 'r', ls = '-', lw = 5)
            if high_lims[nParam] != 0:
                pylab.axhline(y = high_lims[nParam], color = 'r', ls = '-', lw = 5)
            pylab.title(paramNames[nParam])
        pylab.savefig("{0}_params3.png".format(title))
        pylab.close()