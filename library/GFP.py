import numpy as np
import pylab
from sklearn.decomposition import PCA
from heapq import nlargest

import utilities
import segmentationClass

class Sequence():
    def __init__(self, seq, centers, r = 70):
        self.seq = seq
        self.centers = centers
        self.lengthSeq = len(seq)
        self.radiusCell = r
        self.profile = []
        
    def interpolateLinear(self, vals):
        return (vals[0] + vals[1]) /2.
        
    def hScale(self):
        distCenters = (np.max(self.centers) - np.min(self.centers))/2
        center = (self.centers[0] + self.centers[1])/2
        medium = list(self.seq[max(0, center - distCenters * 3): min(center + distCenters *3, self.lengthSeq)])
        if np.sum(self.seq[np.min(self.centers):center])/(center - np.min(self.centers)) > np.sum(self.seq[center:np.max(self.centers)])/(np.max(self.centers) - center):
            medium.reverse()
        if len(medium) != 6*distCenters:
            print "Pb1", len(medium), 6*distCenters, center, self.lengthSeq
            raise IndexError
        if len(medium) > 2*self.radiusCell:
            ## remove len(medium) - 2*self.radiusCell values
            rang = np.asarray(np.linspace(len(medium)-1, 0, num = len(medium)-2*self.radiusCell), dtype = int)
            #print rang, len(medium)
            for i in rang:
                medium.pop(i)
            if len(medium) != 2*self.radiusCell:
                print "Pb2end", len(medium), 2*self.radiusCell, 6*distCenters
                raise IndexError
            ##for the edges we have to remove (len(medium) - 2*r)/len(medium)
            self.profile = np.array(medium)
        elif len(medium) < 2*self.radiusCell :
            ## add 2*self.radiusCell - len(medium)
            out = []
            rang = np.asarray(np.linspace(0, len(medium), num = 2*self.radiusCell - len(medium) + 1, endpoint = False), dtype = int)
            for i in xrange(2*self.radiusCell - len(medium)):
                out.append(self.interpolateLinear(medium[rang[i]:rang[i]+2]))
                out.extend(medium[rang[i]:rang[i+1]])
            out.extend(medium[rang[i+1]:])
            if len(out) != 2*self.radiusCell:
                print "Pb2end", len(medium), 2*self.radiusCell
                raise IndexError
            self.profile = np.array(out)
        else:
            self.profile = np.array(self.seq)
        return self.profile
        
    def vScale(self):
        self.profile = (self.profile - np.percentile(self.profile, 1))/1./np.percentile(self.profile, 99)
        return self.profile
    

class GFPdata():
    def __init__(self, imagespath, outpath = ".", site = -1, extract = (-1, -1), t = range(5), params = []):
        self.imagespath = imagespath
        self.images = []
        self.lengthSeq = len(imagespath)
        for f in imagespath:
            self.images.append(utilities.Extract(f))
        self.outpath = outpath
        self.site = site
        self.extract = extract
        self.t = t
        self.levelGFP = -1
        self.params = params
    
    def getSequence(self):
        out = np.zeros((self.images[0].shape[0], self.images[0].shape[1]*self.lengthSeq))
        for i in xrange(self.lengthSeq):
            out[:, self.images[0].shape[1]*i:self.images[0].shape[1]*(i+1)] = self.images[i].image
        return out           
        
    def getGFP(self):
        m = []
        for im in self.images:
            m.append(im.getsum()/im.shape[0]/im.shape[1])
        m = np.sum(nlargest(3, m))/3.
        self.levelGFP = m
        return m
        
    def getGFPband(self, name = '', time = -1, width = 70, plot = True, save = True):
        results = []
        for index, im in enumerate(self.images):
            dx, dy = np.indices(im.shape)
            c1 = (int(self.params[index][1]), int(self.params[index][2]))
            c2 = (int(self.params[index][1+6]), int(self.params[index][2+6]))
            center = [(self.params[index][1] + self.params[index][1+6])/2., (self.params[index][2] + self.params[index][2+6])/2.]
            product = (dx -c1[0])*(c2[1] - c1[1]) + (dy - c1[1])*(c1[0] - c2[0])
            orthProduct = (dx -c1[0])*(c1[0] - c2[0]) + (dy - c1[1])*(c1[1] - c2[1])
            distance = np.sqrt((dx - c1[0])**2 + (dy - c1[1])**2) * np.sign(orthProduct)
            product = np.abs(product)
            mask = np.zeros(im.shape)
            mask[product < np.percentile(product, 5)] = 1
            X = distance[mask == 1].flatten()
            Y = im.image[mask == 1].flatten()
            Z = zip(X, Y)
            Z.sort()
            Xsort, Ysort = zip(*Z)
            Xsort = np.array(Xsort)
            c1proj = np.argmax(Xsort == 0)
            c2proj = np.argmax(Xsort == distance[c2[0], c2[1]])
            
            try:
                s = Sequence(Ysort, [c1proj, c2proj], r = width)
                s.hScale()
                out = s.vScale()

            except IndexError:
                out = np.zeros(2*width)
                #print "--- debug ---"
            if plot:
                pylab.figure(figsize = (17,6))
                pylab.subplot(1,3,2)
                pylab.imshow(distance*mask)
                pylab.title("Distance")
                pylab.autoscale(False)
                pylab.axis('off')
                pylab.plot(self.params[index][2::6], self.params[index][1::6], 'ro', label = "centres des cellules")
                pylab.legend(numpoints = 1)
                pylab.subplot(1,3,1)
                pylab.title("GFP")
                pylab.imshow(im.image, 'gray')
                pylab.autoscale(False)
                pylab.axis('off')
                pylab.plot(self.params[index][2::6], self.params[index][1::6], 'ro', label = "centres des cellules")
                pylab.legend(numpoints = 1)

                pylab.subplot(1,3,3)
                pylab.plot(Ysort, 'r')
                pylab.axvline(x = c1proj)
                pylab.axvline(x = c2proj)
                pylab.xlabel("Distance")
                pylab.ylabel("Fluorescence (GFP)")

                if save:
                    pylab.savefig("{0}_t{1:02d}.png".format(name, time+index))
                else:
                    pylab.show()
                pylab.close('all')
            results.append(out)
            
        return results, c1, c2
                
        
    def getValueGFP(self, radius = 10, name = '', plot = False):
        mean = []
        movie = np.zeros((self.images[0].shape[0], self.images[0].shape[1]*self.lengthSeq))
        if plot:
            pylab.figure(figsize = (6,22))
        for index, im in enumerate(self.images):
            
            center = [(self.params[index][1] + self.params[index][1+6])/2., (self.params[index][2] + self.params[index][2+6])/2.]
            GFP = segmentationClass.Segmentation(im.image)
            GFP.defineFG()
            GFP.smooth_corr()
            mask = GFP.rond_mask(center, radius)
            val = np.sum(mask*GFP.smooth)/1. /np.sum(mask)
            #val = np.average(GFP.smooth)
            movie[:,index*im.shape[1]:(index+1)*im.shape[1]] = GFP.smooth
            mean.append(val)
            if plot:
                pylab.subplot(len(self.images),2,2*index+1)
                pylab.imshow(GFP.smooth, 'gray')
                #pylab.title("t = {0}".format())
                pylab.axis('off')
                pylab.subplot(len(self.images),2,2*index+2)
                pylab.imshow(mask, 'gray')
                pylab.autoscale(False)
                pylab.axis('off')
                pylab.plot(self.params[index][2::6], self.params[index][1::6], 'ro')
                pylab.title('{0:.0f}'.format(val))
        if plot:   
            pylab.savefig(name)
            pylab.close()
        return np.sum(nlargest(3, mean))/3., movie
        
    