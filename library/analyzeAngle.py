import pylab
import os
from glob import glob
from random import random, randint, sample
import numpy as np
import scipy.stats
import subprocess
from heapq import nlargest
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

import utilities
import segmentationClass

class ResultFile():
    def __init__(self, filepath):
        self.filepath = filepath
        self.errorDecision = False
        self.distDaughterDecision = False
        self.orderDecision = False
        self.site = -1
        self.extract = [-1, -1]
        self.t = -1
        self.angle = -1.
        self.readAndExtract()

    def readAndExtract(self):
        for line in open(self.filepath, 'r').read().split('\n'):
            if len(line) > 0:
                if line[:4] == "dist":
                    self.distDaughterDecision = eval(line.split(':')[-1])
                elif line[:4] == "Dist" :
                    self.orderDecision = eval(line.split(',')[-1].lstrip(' rightOrder'))
                elif line[:3] == ">>>" :
                    contents = line[3:].split(',')
                    self.site = eval(contents[0].lstrip('SITE0'))
                    self.extract = contents[1].lstrip('EXTRACT0').split('&')
                    self.extract = [eval(self.extract[0]), eval(self.extract[1].lstrip('0'))]
                    self.t = eval(contents[2].lstrip('T'))
                    self.angle = eval(contents[3].lstrip('ANGLE'))


class interactivePlot():
    def __init__(self, X, Y, dataObject, imagespath):
        self.X = X
        self.Y = Y
        if len(X) != len(Y):
            raise IndexError("X and Y don't have the same length.")
        self.lengthSeq = len(X)
        self.dataObject = dataObject
        self.stats = np.ones(self.lengthSeq)*-1
        self.imagespath = imagespath
        self.GFP = []
        self.plot()

    def plot(self):
        colors = np.array(['r', 'y', 'g', 'c', 'm', 'b'])
        cat = ['NT, 500ng/mL DOX', 'DLG siRNA, 500ng/mL DOX', 'NuMA siRNA, 500ng/mL DOX', 'NT, 1ug/mL DOX']
        bins = [np.percentile(self.dataObject.GFP, 100*i/len(colors)) for i in range(len(colors))]
        print bins
        bins = np.array(bins)
        colorsCat = np.array([np.argmin(bins<i) for i in self.dataObject.GFP])
        print colorsCat[:25]
        self.GFPcats = colorsCat
        fig = pylab.figure()
        self.ax = fig.add_subplot(111)
        scat = self.ax.scatter(self.X, self.Y, c = colors[colorsCat], picker = 5, alpha = .7)
        fig.canvas.mpl_connect('pick_event', self.onpick)
        pylab.xlabel('Categories')
        pylab.ylabel('Reference Angle (degree)')
        pylab.title('500ng/mL DOX : 0, NT; 1, DLG siRNA; 2, NuMA siRNA. 1ug/mL DOX: 3, NT')
        pylab.show()

    def onpick(self, event):
        for dataind in event.ind:
            site = self.dataObject.sites[dataind]
            if len(str(site)) == 1:
                site = '0'+str(site)
            x1, y1 = self.dataObject.extracts[dataind]
            t = self.dataObject.times[dataind]
            f = glob( '{0}ANGLEsite{1}_extract{2}&{3}_t{4}.png'.format(self.imagespath, int(site), x1, y1, t))
            if len(f) >1 :
                print f
            elif len(f) == 0:
                print '{0}ANGLEsite{1}_extract{2}&{3}_t{4}.png'.format(self.imagespath, int(site), x1, y1, t)
            subprocess.call(['eog', f[0]], stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
            val = input('Good (1) or not (0) ? If you want the stats, enter a negative number. ')
            self.stats[dataind] = val
            while val < 0 :
                print "True : {0}, False {1}".format(len(self.stats[self.stats==1]), len(self.stats[self.stats==0]))
                val = input('Good or not ? ')
            if val == 0:
                self.ax.scatter(self.X[dataind], self.Y[dataind], c= 'k', marker = 'v')
                print "!!! BUG : ", f[0]
            else:
                self.ax.scatter(self.X[dataind], self.Y[dataind], c = 'r', marker = 'v')

            pylab.draw()
        return True



class data():
    cat = ['NT, 500ng/mL DOX', 'DLG siRNA, 500ng/mL DOX', 'NuMA siRNA, 500ng/mL DOX', 'NT, 1ug/mL DOX']
    colors = ['r', 'y', 'g', 'c', 'm', 'b']

    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.files = glob("{0}*result".format(dirpath))
        self.files.sort()
        self.dataQty = len(self.files)
        self.seq = []
        self.sites, self.extracts, self.times, self.angles, self.categories = [], [], [], [], []
        self.GFP = []

    def getData(self):
        for f in self.files:
            fileobj = ResultFile(f)
            fileobj.readAndExtract()
            self.seq.append(fileobj)
            if fileobj.t != -1 and fileobj.distDaughterDecision and fileobj.orderDecision:
                self.sites.append(fileobj.site)
                self.extracts.append(fileobj.extract)
                self.times.append(fileobj.t)
                self.angles.append(fileobj.angle)
                self.categories.append(self.returnGroup(fileobj.site))
        self.angles = np.asarray(self.angles)*180/np.pi
        self.sites = np.asarray(self.sites)
        self.times = np.asarray(self.times)
        self.extracts = np.asarray(self.extracts)
        self.categories = np.asarray(self.categories)

    def plot(self, imagespath, toplotX = "categories", toplotY = "angles"):
        if toplotX == "categories":
            X = self.categories + np.random.randn(len(self.categories))/10
        if toplotY == "angles":
            Y = self.angles
        p = interactivePlot(X, Y, self, imagespath)

    def histplot(self, extradataA = [], extradataG = [], intensity = []):
        pylab.figure(figsize = (25,8))
        cat = ['NT, 500ng/mL DOX', 'DLG siRNA, 500ng/mL DOX', 'NuMA siRNA, 500ng/mL DOX', 'NT, 1ug/mL DOX']
        pops = []
        for i in xrange(3):
            pylab.subplot(1,3,i+1)
            pop = self.angles[(self.categories == i)]# &  (self.GFP > -np.log(12.5))]# & (intensity == 'r')]
            print "cat {0}, pop {1}, pop + GFP {2}".format(i, len(self.angles[self.categories == i]), len(pop))
            pops.append(pop)
            hist, binedges = np.histogram(pop, bins = 18)
            pylab.tick_params(axis='both', which='major', labelsize=25)
            pylab.plot(binedges[:-1], np.cumsum(hist)/1./len(pop), data.colors[i], label = data.cat[i], linewidth = 4)
            if len(extradataA) > i:
                print extradataA[i]
                h, bins = np.histogram(extradataA[i], bins= 18)
                hbis = h/1./len(extradataA[i])
                x, y = [], []
                for index in xrange(len(hbis)):
                    x.extend([bins[index], bins[index+1]])
                    y.extend([hbis[index], hbis[index]])
                print x, y, len(x)
                pylab.tick_params(axis='both', which='major', labelsize=25)
                pylab.plot(bins[:-1], np.cumsum(h)/1./len(extradataA[i]), 'k', linewidth = 4)

            pylab.xlabel("Angle (degre)", fontsize = 25)
            #pylab.title(cat[i])
            pylab.ylim([0., 1.2])
            pylab.legend(loc = 2, prop = {'size' : 20})
        for ip, p in enumerate(pops):
            for ip2, p2 in enumerate(pops):
                ksstat, kspval = scipy.stats.ks_2samp(p2, p)
                print "#### cat{0} & cat{3} : ks Stat {1}, pvalue {2}".format(ip, ksstat, kspval, ip2)
        pylab.show()
        #pylab.savefig("{0}hist.png".format(dirpath, nbins, 2, randint(0,999), dirpath))

    def plotAgainstGFP(self, extradataA = [], extradataG = [], intensity = [], seq = []):
        fig1 = pylab.figure(figsize = (25, 10))
        print len(self.GFP)
        for i in xrange(min(len(data.cat), 3)):
            print len(self.GFP[self.categories == i])
            vect = []
            pylab.subplot(1,3,i+1)
            #pylab.hist(self.GFP[self.categories == i], bins = 20, color = data.colors[i])
            pop = self.GFP[self.categories == i]
            pylab.plot(self.GFP[self.categories == i], self.angles[self.categories == i], data.colors[i]+'o', markersize = 8)#, label = data.cat[i])
            print "cat", i, "n pop", len(self.GFP[(self.categories == i) & (self.GFP > -np.log(12.5))])
            x = np.linspace(np.min(self.GFP[self.categories == i]), np.percentile(self.GFP[self.categories == i], 80),40)
            #fig1.canvas.mpl_connect('pick_event', onpick)
            for j in x:
                vect.append(np.median(self.angles[(self.GFP > j) & (self.categories == i)]))

            pylab.plot([-4.5, -0.5], [vect[0], vect[0]], data.colors[i], label = "mediane de la population entiere", linewidth = 5)
            print vect[0], vect[np.argmax(x > -np.log(12.5))]
            pylab.plot([-np.log(12.5), -0.5], [vect[np.argmax(x > -np.log(12.5))] for k in  [0,1]], data.colors[i], label = "mediane de la population de droite", linewidth = 5, ls = '--')
            pylab.axvline(x = -np.log(12.5), color = 'm', ls = '--', linewidth = 3)
            pylab.xlim([-4.5, -0.5])
            pylab.legend(loc = 2, prop = {'size':17})

            pylab.title(data.cat[i].split(',')[0], fontsize = 24)
            pylab.xlabel('score GFP', fontsize = 20)
            pylab.ylabel('Angle (degre)', fontsize = 20)
            pylab.tick_params(axis='both', which='major', labelsize=20)
            pylab.ylim([-5, 105])
            ##pylab.xscale('log')
        pylab.show()


    def plotAgainstGFP_hist2d(self):
        fig1 = pylab.figure(figsize = (20, 15))
        print len(self.GFP)
        for i in xrange(min(len(data.cat), 4)):
            print len(self.GFP[self.categories == i])
            vect = []
            pylab.subplot(2,2,i+1)
            pop = self.GFP[self.categories == i]
            print "cat", i, "n pop", len(self.GFP[(self.categories == i) & (self.GFP > -np.log(12.5))])
            H, xedges, yedges = np.histogram2d(self.angles[self.categories == i], self.GFP[self.categories == i], bins = 10)
            hist = pylab.hist2d(self.GFP[self.categories == i], self.angles[self.categories == i], bins = 10, cmap = pylab.cm.Reds, normed = True)
            pylab.clim(0.,0.035)
            pylab.colorbar()
            pylab.title(data.cat[i])
            pylab.xlabel('GFP score')
            pylab.ylabel('Angle (degree)')
            pylab.xlim([-4.2, -1])
        pylab.show()


    def ttest(self):
        angles = [self.angles[(self.categories == i) & (self.GFPcats > 0)] for i in xrange(len(data.cat))]
        print angles, self.GFPcats
        for i in xrange(len(data.cat)):
            for j in xrange(i+1, len(data.cat)):
                statT, pvalue = scipy.stats.ttest_ind(angles[i], angles[j], equal_var=False)
                print "cat{0} & cat{1} get {2} ({3})".format(i,j, pvalue, statT)


    def MWUtest(self):
        for i in xrange(len(data.cat)):
            for j in xrange(i+1, len(data.cat)):
                statT, pvalue = scipy.stats.mannwhitneyu(self.angles[(self.categories == i) ], self.angles[(self.categories == j)])
                print "cat{0} & cat{1} get {2} ({3})".format(i,j, pvalue, statT)

    def MWUtest_extradata(self, extradata):
        for i in xrange(len(extradata)):
            for j in xrange(i+1, len(extradata)):
                statT, pvalue = scipy.stats.mannwhitneyu(extradata[i], extradata[j])
                print "cat{0} & cat{1} get {2} ({3})".format(i,j, pvalue, statT)

    def bootstrap(self, nBoot, nbins = 20):
        pops = np.zeros((nBoot, nbins))
        #medianpop = [[] for i in data.cat]
        pylab.figure(figsize = (20,14))
        for i in xrange(3):
            pylab.subplot(1,3,i+1)
            #if  i ==0:
                #pylab.title("Bootstrap on medians", fontsize = 20.)
            pop = self.angles[(self.categories == i)]# & (self.GFP > 2000)]
            for index in xrange(nBoot):
                newpop = np.random.choice(pop, size=len(pop), replace=True)
                #medianpop[i].append(np.median(newpop))
                newhist, binedges = np.histogram(newpop, bins = nbins)
                pops[index,:] = newhist/1./len(pop)
            #pylab.hist(medianpop[i], bins = nbins, label = "{2} median {0:.1f}, std {1:.1f}".format(np.median(medianpop[i]), np.std(medianpop[i]), data.cat[i]), color = data.colors[i], alpha =.2, normed = True)

            meanpop = np.sum(pops, axis = 0)/1./nBoot
            stdY = np.std(pops, axis = 0)
            print "width", binedges[1] - binedges[0]
            pylab.bar(binedges[:-1], meanpop, width = binedges[1] - binedges[0], label = "mean distribution", color = data.colors[i], alpha = 0.6)
            pylab.fill_between((binedges[:-1]+binedges[1:])/2., meanpop-stdY, meanpop+stdY, alpha = 0.3)
            pylab.legend()
            pylab.title(data.cat[i])
            pylab.xlabel("Angle(degree)", fontsize = 15)
            pylab.ylim([-.01, 0.23])

        pylab.savefig("/users/biocomp/frose/frose/Graphics/FINALRESULTS-diff-f3/distrib_nBootstrap{0}_bins{1}_GFPsup{2}_{3}.png".format(nBoot, nbins, 'all', randint(0,999)))

    def bootstrap_extradata(self, nBoot, extradataA, nbins = 20):
        pops =[]
        meanpop = [[] for i in data.cat]
        pylab.figure(figsize = (14,14))
        for i in xrange(min(4, len(extradataA))):
            #pylab.subplot(2,2,i+1)
            if  i ==0:
                pylab.title("Bootstrap on means", fontsize = 20.)
            pop = extradataA[i]# & (self.GFP > 2000)]#
            for index in xrange(nBoot):
                newpop = np.random.choice(pop, size=len(pop), replace=True)

                #meanpop[i].append(np.mean(newpop))
            pops.append(newpop)
            pylab.legend()
        #pylab.title(cat[i])
            pylab.xlabel("Angle(degree)", fontsize = 15)
            pylab.xlim([0., 90.])
        for i in xrange(len(extradataA)):
            for j in xrange(i+1, len(extradataA)):
                statT, pvalue = scipy.stats.ttest_ind(pops[i], pops[j], equal_var=False)
                print "cat{0} & cat{1} get {2} ({3})".format(i,j, pvalue,statT)
        pylab.savefig("/users/biocomp/frose/frose/Graphics/FINALRESULTS-diff-f3/mean_nBootstrap{0}_bins{1}_GFPsup{2}_FLO_{3}.png".format(nBoot, nbins, 'all', randint(0,999)))

    @staticmethod
    def returnGroup(site):
        if site <= 21:
            return 1
        elif site <= 40:
            return 2
        elif site <= 58:
            return 0
        else:
            return 3

    def addGFP(self, vect):
        self.GFP = np.array(vect)

