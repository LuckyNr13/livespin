#!/usr/bin/env python

import cv2.cv as cv
import cv2
import pylab
import scipy.ndimage as ndimage
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

   
class ExtractImage():
    def __init__(self, image, gauss = 2):
        """

        :rtype : object
        """
        self.image = np.array(image)
        self.shape = self.image.shape
        self.smooth = self.getSmooth(gauss)
    
    def hist2d(self, title, newfig = True):
        if newfig :
            fig1 = pylab.figure()
            ax = Axes3D(fig1)
        else:
            ax = Axes3D(newfig)
        dx, dy = np.indices(self.shape)
        ax.plot_wireframe(dx, dy, self.image, alpha = 0.2)
        pylab.title(title)
        return ax
    
    def rond_mask(self, center, radius):
        ker = np.zeros(self.shape)
        for i in xrange(self.shape[0]):
            for j in xrange(self.shape[1]):
                d = (center[0]-i)**2 + (center[1]-j)**2
                if d <= radius**2:
                    ker[i,j] = 1
        return ker
 
    def display(self):
        for line in self.image:
            print " , ".join([str(int(i)) for i in line])
    
    def getSmooth(self, gauss = 2):
        return ndimage.filters.gaussian_filter(self.image, gauss)
 
    def max_filter(self, window):
        size = len(self.image[0])
        maxi = np.zeros(np.shape(self.image), dtype = np.uint8)
        window = int(window)
        for dx in xrange(window/2, size-window/2):
            for dy in xrange(window/2, size-window/2):
                px = self.image[dx,dy]
                s = np.sum((self.image[dx-window/2:dx+window/2, dy-window/2:dy+window/2] >= px))
                if s <=  1:
                    maxi[dx,dy] = 1
        return maxi
    
    def convert16to8(self, smooth = False):
        if smooth:
            matout = self.smooth
        else:
            matout = self.image
        p = np.percentile(matout, 98)
        lowp = np.percentile(matout, 1)
        matout -= lowp
        if p == 0:
            p = np.max(matout)
        #print lowp, p, np.max(matout)
        matout[matout < 0] = 0
        matout[matout > p] = p
        #pylab.imshow(matout/p*255, 'gray')
        #pylab.show()
        return np.array(matout/p*255, np.uint8)
    
    def getsum(self):
        return np.sum(self.image)
    
    def removeIllumination(self, size, title = ''):
        min_ = ndimage.filters.minimum_filter(self.image, size)
        out = self.image - min_
        print "oui"
        
        mingauss = ndimage.filters.gaussian_filter(min_, size/10.)
        minout = ndimage.filters.minimum_filter(out, size)
        pylab.figure()
        pylab.subplot(231)
        pylab.imshow(self.image, 'gray')
        pylab.axis('off')
        pylab.subplot(232)
        pylab.imshow(min_, 'gray')
        pylab.axis('off')
        pylab.subplot(233)
        print np.max(out), np.min(out)
        pylab.imshow(out, 'gray')
        pylab.axis('off')
        pylab.subplot(234)
        pylab.imshow(minout, 'gray')
        pylab.axis('off')
        pylab.subplot(235)
        pylab.imshow(mingauss, 'gray')
        pylab.axis('off')
        pylab.subplot(236)
        print np.max(self.image-mingauss), np.min(self.image-mingauss)
        pylab.imshow(self.image-mingauss, 'gray')
        pylab.axis('off')
        print "/users/biocomp/frose/frose/Graphics/honeycomb_grid/{0}.png".format(title)
        return out
        #pylab.show()
        
    def removeIllumination2(self, size, title = ''):
        out = ndimage.filters.gaussian_filter(self.image, size)
        pylab.figure()
        pylab.subplot(2,2,1)
        pylab.axis('off')
        pylab.imshow(self.image)
        pylab.subplot(2,2,2)
        pylab.axis('off')
        pylab.imshow(out)
        pylab.subplot(2,2,3)
        pylab.axis('off')
        pylab.imshow(self.image - out)
        pylab.subplot(2,2,4)
        pylab.axis('off')
        pylab.imshow(self.smooth - out)
        if title != '':
            pylab.savefig(title)
            pylab.close()
        else:
            pylab.show()
        self.smooth -= out
        return self.image - out
        
class Extract(ExtractImage):
    def __init__(self, imagepath):
        self.imagepath = imagepath
        ExtractImage.__init__(self, self.open(self.imagepath))

        
    def open(self, imagepath):
        im = cv2.imread(imagepath, cv.CV_LOAD_IMAGE_UNCHANGED)
        if im == None:
            raise Exception('No file found at this path :{0}'.format(imagepath))
        else:
            if im.shape == (100, 100, 3):
                im = im[:,:,0] +im[:,:,1]+im[:,:,2]
        return im
        

    
    