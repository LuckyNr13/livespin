# This code is to extract and crop image sequence in each micropattern from the pattern media images by using template image.
# author: Yingbo Li, Auguste Genovesio

#! /users/biocomp/yb/envlyb/bin python

__author__ = 'yb'


import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
from glob import glob
import argparse
from sys import argv

import sys

sys.path.append('library/')
import tifffile as tiff
import tiffcapture as tc
from skimage import exposure
import cv2
from skimage.measure import label
from skimage.measure import regionprops
from scipy.signal import argrelmax
from skimage.morphology import erosion
from skimage.morphology import disk
import library.segmentationClass as segmentationClass
import library.utilities as utilities


#######################################################################################
## HELP AND INPUT ARGUMENTS
################################################################################54#############
parser = argparse.ArgumentParser(description = 'Extract the individual pattern with cells and create extract directories with the cropped images inside. It will work on the mCherry images, those correspoding to laser 561 nm. Parameters with "CONDOR" in it are for internal functionning of the script: do not use those.', formatter_class = argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-I', '--IN', type = str, help = "path to the original images. No default argument. Needed. (str)", dest = "path")
parser.add_argument('-o', '--out', dest = 'out', type = str, default = 'result', help = "appendix name that will be add to the folder name to indicate it contains the results. (str)")
parser.add_argument('-cd', '--c-to-decide', type = str, help = "channel to take the decision, i.e. name of the channel corresponding to the nuclei images. keyword. (str)", dest = 'decide', default = 'w1')
parser.add_argument('-ce', '--c-to-extract', nargs = '+',type = str, help = "channel(s) considered for the extraction part. keyword or 'all'. All the images files with this keyword will be submitted to extraction. It is a time consuming operation, so if you do not need the small images for the other channels, you may want to precise only the nuclei channel with this option. (str)", dest = 'extract', default = 'all')
parser.add_argument('-c', '--pathpre-to-extract', type = str, help = "the file beginning with it is extracted", dest = 'pathpre', default = 'edga_')


parser.add_argument('-ng', '--withoutGrid', dest='withGrid',action='store_false', help = "Activate this option if you want to detect patterns only by searching the maxima. (too tilted grid or grid of another form - honeycomb) Default is with grid detection.")
parser.set_defaults(withGrid=True)

parser.add_argument('-n', '--nImages', type = int, help = "number of images that will be averaged to decide where cells are. The last images are taken. (int)", dest = 'nImages', default = 100)
parser.add_argument('-g', '--gauss', type = int, help = "size of the gaussian kernel for smoothing the averaged image. (int)", dest = 'gaussian', default = 8)
parser.add_argument('-d', '--dist', type = int, help = "distance between sites. (int, in pixels)", dest = 'dist', default = 100)


parser.add_argument('-t', '--test',dest='test',action='store_true', help = "If test, the script only detects the sequences to extract but does not extract the images (time consuming). To use if you are not sure about the parameters. It will create and save the averaged images.")
parser.set_defaults(test=False)

parser.add_argument('-db', '--debug', dest='debug',action='store_true', help = "If the debug parameter is activated, it creates the stderr and stdout files of the condor jobs.")
parser.set_defaults(debug=False)


parser.add_argument('-C', '--CONDOR',dest='CONDOR',action='store_true', help = "DO NOT USE")
parser.set_defaults(CONDOR=False)
parser.add_argument('-nC', '--nSite-CONDOR', type = int, help = "site number. DO NOT USE. (int)", dest = 'nsite', default = 0)

args = parser.parse_args(argv[1:])
if args.path == None:# if donot have input path, use this
    args.path = "/users/biocomp/yingboli/livespin/3-11-15/Experiment_5-3-15-plate_3-examples/B2-LGNsi"
    args.gaussian = 4

print args
##############################################################################################

def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


if argv[0][0] == '/':
    path_script = '/'.join(argv[0].split('/')[:-1])
else:
    path_script = argv[0].split('/')[:-1]
    if len(path_script) == 1:
        path_script = '/'.join([os.getcwd(), path_script[0]])
    elif len(path_script) == 0:
        path_script = os.getcwd()
    else:
        content = [os.getcwd()]
        content.extend(path_script)
        path_script = '/'.join(content)

os.chdir(args.path)
listfiles_all_channel = glob('{0}/{1}*.tif'.format('cy5 initial', args.pathpre))
if len(listfiles_all_channel) == 0:
    listfiles_all_channel = glob('{0}*.TIF'.format(args.pathpre))

if len(listfiles_all_channel) == 0:
    raise Exception("Wrong value of '--c-to-decide' parameter : {0}".format(args.decide))


print os.getcwd()
listfiles_all_channel.sort()

## prepare for extraction

if args.path[-1] == '/':
    path = args.path[:-1]
else:
    path = args.path

outpath = '{0}/{1}_{2}'.format(path, path.split('/')[-1], args.out)
if not os.path.isdir(outpath):
    os.mkdir(outpath)

## iteration on the observation sites (labeled with s + number)
nSite = 1
listfiles=listfiles_all_channel

# process cy5 initial in order to get the positions of patterns,also the size of windows and its area size
pos_all=[]
areas_all=[]
centers_all=[]
pattern_arm_all=[]
for i in xrange(len(listfiles)):
    patt = tc.opentiff(path+'/'+listfiles[i])
    temp = patt.find_and_read(0)
    imObj = utilities.ExtractImage(temp, 6)
    img_adapteq = exposure.equalize_adapthist(imObj.smooth, clip_limit=0.03)
    img_adapteq /= (np.max(img_adapteq)/255.)
    hist,bins = np.histogram(img_adapteq,256,[0,256])
    hist2=smooth(hist,40)
    histmax = argrelmax(hist2)
    ret, thresh=cv2.threshold(np.array(img_adapteq, dtype = np.uint8), (histmax[-1][-1]+255)/2, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    thresh[thresh == 0] = 1
    thresh[thresh == 255] = 0
    selem = disk(5)
    thresh = erosion(thresh, selem)
    label_image = label(thresh)
    rep = regionprops(label_image)

    centers = np.zeros((len(rep),2))
    areas = np.zeros(len(rep))
    for j in xrange(len(rep)):
        centers[j, :] = rep[j]['centroid']
        areas[j] = rep[j]['area']
    pattern_arm = np.round((np.percentile(areas, 95)/np.pi)**0.5+5)
    centers = np.delete(centers, np.concatenate([np.where(areas < np.median(areas)/4.)[0],np.where(areas > np.median(areas)*3.)[0]]), axis=0)
    areas = np.delete(areas, np.concatenate([np.where(areas < np.median(areas)/4.)[0],np.where(areas > np.median(areas)*3.)[0]]))
    areas = np.delete(areas,  np.concatenate([np.where(centers[:, 0] < pattern_arm*2)[0], np.where(centers[:, 1] < pattern_arm*2)[0], np.where(thresh.shape[0]-centers[:, 0] < pattern_arm*2)[0], np.where(thresh.shape[1]-centers[:, 1] < pattern_arm*2)[0]]))
    centers = np.delete(centers,  np.concatenate([np.where(centers[:, 0] < pattern_arm*2)[0], np.where(centers[:, 1] < pattern_arm*2)[0], np.where(thresh.shape[0]-centers[:, 0] < pattern_arm*2)[0], np.where(thresh.shape[1]-centers[:, 1] < pattern_arm*2)[0]]), axis=0)


    areas_all.append(areas)
    centers_all.append(centers)
    pattern_arm_all.append(pattern_arm)
    pos_all.append(int(listfiles[i].split('Pos')[1].split('.')[0]))

data = {'pos': pos_all, 'areas': areas_all, 'center': centers_all, 'pattern_radius': pattern_arm_all}
np.save(outpath+'/splittingdata', data)

## extract cropped images and save them in new directories
listvideo = glob('{0}*.tif'.format(args.pathpre))
listvideo.sort()
for m in xrange(len(listvideo)):
    posv = int(listvideo[m].split('Pos')[1].split('.')[0].split('_')[0])
    ind1 = data['pos'].index(posv)
    armsize = data['pattern_radius'][ind1]*1.8
    centers = np.round(data['center'][ind1])
    centers = centers.astype(np.int)

    pos_dir = 'Pos' + listvideo[m].split('.ome')[0].split('Pos')[-1]
    print '-- Extracting {0} video --'.format(m)
    tiffimg = tc.opentiff(path+'/'+listvideo[m])

    if len(pos_dir.split('_')) == 2:
        xi = 0
    else:
        xi = 1

    bllist =[]
    minint=np.zeros([tiffimg.length/2, len(centers)])
    maxint=np.zeros([tiffimg.length/2, len(centers)])
    #
    for n in range(xi, tiffimg.length, 2):
        image = tiffimg.find_and_read(n)
        if n % 2 == 0:
            ind2=n/2
        else:
            ind2=(n-1)/2
        print "#### EXTRACTING WINDOWS -- {0:.2f} ({1} video)% processed".format(float(n+1)/tiffimg.length*100, m)
        for j in xrange(len(centers)):

            if j in bllist:
                continue
            name_dir = "dir_x{0:04d}_y{1:04d}".format(centers[j,0], centers[j,1])

            if len(image.shape) == 3:
                ext = image[centers[j,0]-armsize:centers[j,0]+armsize, centers[j,1]-armsize:centers[j,1]+ armsize, :]
            else:
                ext = image[centers[j,0]-armsize:centers[j,0]+armsize, centers[j,1]-armsize:centers[j,1]+ armsize]
            #ext_temp = (ext-minint[j])*1./(maxint[j]-minint[j])*255.
            ext_temp = (ext-np.min(ext))*1./(np.max(ext)-np.min(ext))*255.
            ext_temp = np.array(ext_temp)
            ext_temp = ext_temp.astype(np.uint8)

            #test if blank image
            if n==xi:
                seg = segmentationClass.Segmentation(ext_temp, gauss = 1, gaussMask = 2, size = 70, maxFilter = 12)
                seg.findMaximaOnFG([])
                if seg.areamax > 15.**2*np.pi*2.5 or np.sum(seg.FG)>15.**2*np.pi*3 or seg.areanum > 3 or seg.areanum==0:
                    bllist.append(j)
                else:
                    os.chdir(outpath)
                    if not os.path.isdir(pos_dir):
                        os.mkdir(pos_dir)
                    os.chdir(outpath+'/'+pos_dir)
                    if not os.path.isdir(name_dir):
                        os.mkdir(name_dir)
                    os.chdir(outpath+'/'+pos_dir+'/'+name_dir)
                    if not os.path.isdir('images'):
                        os.mkdir('images')

                    tiff.imsave("images/{0}_x{1:04d}_y{2:04d}_t{3:04d}.tif".format(listvideo[m].split('.')[0], centers[j,0], centers[j,1], ind2), ext)
            #
            else:
                os.chdir(outpath+'/'+pos_dir+'/'+name_dir)

                tiff.imsave("images/{0}_x{1:04d}_y{2:04d}_t{3:04d}.tif".format(listvideo[m].split('.')[0], centers[j,0], centers[j,1], ind2), ext)

