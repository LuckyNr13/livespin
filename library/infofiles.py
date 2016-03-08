import pylab, matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
from time import time


class InfoFile():
    def __init__(self, filepath, site, x, y):
        self.filepath = filepath
        self.site = site
        self.x = x
        self.y = y
        self.f2, self.f3 = self.getError()
        self.f2, self.f3, self.para2, self.para3 = self.getEandP()
        self.lengthSeq = len(self.f2)
            
    def getError(self):
        funVal = []
        for line in open(self.filepath, 'r').read().split('\n'):
            if len(line) > 0 and line[0] != '#' and line[0] != 'N':
                c = eval(line.split(',')[-1])
                funVal.append(c)
        funVal = np.array(funVal)
        f2 = funVal[::2]
        f3 = funVal[1::2]
        return f2, f3

    def getEandP(self):
        funVal = []
        para2 = []
        para3 = []
        for line in open(self.filepath, 'r').read().split('\n'):
            if len(line) > 0 and line[0] != '#' and line[0] != 'N':
                c = eval(line.split(',')[-1])
                funVal.append(c)
                if line.split(',')[0] == '2':
                    para2.append(line.split(',')[1:-1])
                if line.split(',')[0] == '3':
                    para3.append(line.split(',')[1:-1])
        funVal = np.array(funVal)
        para2 = np.array(para2)
        para3 = np.array(para3)
        f2 = funVal[::2]
        f3 = funVal[1::2]
        return f2, f3, para2, para3
        
    def getGaussians(self, t):
        params2, params3 = [], []
        lineBool = False
        for line in open(self.filepath, 'r').read().split('\n'):
            if len(line) > 0:
                if not lineBool:
                    if line[:4] == '####':
                        # tline = eval(line.split('_t')[-1].split('.')[0].split('_')[0].lstrip('0'))
                        tline = line.split('_t')[-1].split('.')[0].split('_')[0]
                        if type(t) == list:
                            if tline in t:
                                lineBool = True
                        else:
                            if int(tline) == t:
                                lineBool = True
                else:
                    try:
                        if eval(line[0]) == 2:
                            params2.append([eval(p) for p in line.split(',')[1:-1]])
                        elif eval(line[0]) == 3:
                            params3.append([eval(p) for p in line.split(',')[1:-1]])
                    except SyntaxError:
                        if line[:4] == '####':
                            lineBool = False
                        else:
                            pass
        return params2, params3
        
    def getAllGaussians(self):
        params2, params3 = [], []
        for line in open(self.filepath, 'r').read().split('\n'):
            if len(line) > 0:
                try:
                    if eval(line[0]) == 2:
                        params2.append([eval(p) for p in line.split(',')[1:-1]])
                    elif eval(line[0]) == 3:
                        params3.append([eval(p) for p in line.split(',')[1:-1]])
                except (SyntaxError, NameError):
                    pass
        return params2, params3
        
    def plot(self, outpath=''):
        pylab.figure(figsize = (17,10))
        diff = self.f2-self.f3
        pylab.subplot(2,1,1)
        pylab.plot(range(self.lengthSeq), self.f2, 'r-', label = "f2")
        pylab.plot(range(self.lengthSeq), self.f3, 'g-', label = "f3")
        pylab.xlim([0., self.lengthSeq])
        pylab.tick_params(axis='both', which='major', labelsize=25)
        pylab.subplot(2,1,2)

        diff2 = diff/self.f3
        diff2 /= np.max(diff2)
        pylab.plot(range(self.lengthSeq), diff2, 'b-', label = "Rescaled (by max) difference / f3")
        pylab.xlabel("Temps (en images)", fontsize = 25)
        pylab.tick_params(axis='both', which='major', labelsize=25)
        pylab.xlim([0., self.lengthSeq])
        #pylab.legend(loc= 2, prop = {'size':15})
        pylab.savefig(outpath)
        pylab.close()
        
    def diff(self):
        diff = (self.f2-self.f3)/self.f3
        diff /= np.max(diff)
        return diff
        
class ExtractDirectory():
    def __init__(self, dir_name):
        self.name = dir_name
        self.site, self.x, self.y = self.getInfo()
    
    def getInfo(self):
        site = self.name.split('_s')[1].split('_')[0]
        x = self.name.split('_x')[1].split('_')[0].lstrip('0')
        y = self.name.split('_y')[1].lstrip('0')
        return int(site), int(x), int(y)