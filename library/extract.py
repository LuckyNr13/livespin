#! /usr/bin/env python

import cv
import cv2
import pylab
import os
import numpy as np

import utilities 


def save_maxima(image, new_x, new_y, dist, outpath, nameCondition):
    pylab.close('all')
    pylab.imshow(image)
    pylab.axis('off')
    pylab.autoscale(False)
    for i in xrange(len(new_x)):
        points = pylab.plot(new_x[i], new_y[i], 'wo')
        squares = pylab.gca().add_patch(pylab.Rectangle((new_x[i] - dist/2, new_y[i] - dist/2), dist, dist, edgecolor = 'w', alpha = 0.3, lw = 3))
    pylab.savefig("{0}/found_maxima_n{1}_{2}.png".format(outpath, len(new_x), nameCondition))
    pylab.close()
     
    
def extract_and_save(listdir, site, x, y, path, outpath, size_im, frame_size):
    print '-- Extracting {0} sites --'.format(len(x))  
    for i in xrange(len(x)):
        print "#### EXTRACTING WINDOWS -- {0:.2f}% processed".format(float(i+1)/len(x)*100.)
        name_dir = "dir_s{0}_x{1:03d}_y{2:03d}".format(site, x[i], y[i])
        os.chdir(outpath)
        if not os.path.isdir(name_dir):
            os.mkdir(name_dir)
        os.chdir(outpath+'/'+name_dir)
        if not os.path.isdir('images'):
            os.mkdir('images')
        for n, image_file in enumerate(listdir): 
            os.chdir(path)
            image = utilities.Extract(image_file).image
            ymin = max(y[i]-frame_size/2, 0)
            xmin = max(x[i]-frame_size/2, 0)
            if len(np.shape(image)) == 3:
                ext = image[ymin:ymin+frame_size, xmin:xmin+frame_size, :]
            else:
                ext = image[ymin:ymin+frame_size, xmin:xmin+frame_size]
            os.chdir(outpath+'/'+name_dir)
            if ext.shape != (frame_size, frame_size):
                print "Problem : the extracted image has not the right shape : {0}, t = {1}".format(name_dir, n+1)
            try :
                cv2.imwrite("images/{0}_x{1:03d}_y{2:03d}".format(image_file.split('.')[0], x[i], y[i])+".tif", ext)
            except Exception:
                print "{0}_x{1:03d}_y{2:03d}".format(image_file.split('.')[0], x[i], y[i])+".tif"
                print "Problem when saving the extracted image : {0}, t = {1}".format(name_dir, n+1)
          
