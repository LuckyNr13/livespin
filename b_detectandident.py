# For image sequence of mitosis in each micropattern, this code could automatically identify the division time and angle
# from 2 cells to 3 cells
# author: Yingbo Li, France Rose, Auguste Genovesio

#! /import/bc_users/biocomp/yingboli/envlyb/bin python
__author__ = 'yb'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pylab
from glob import glob
import argparse
from sys import argv
import csv
#############################################################################################
## HELP AND INPUT ARGUMENTS
#############################################################################################
parser = argparse.ArgumentParser(
    description='Script to fit on each image the gaussian mixture with 2 and 3 components, then detection division and compute division angle. '
                'There are general arguments and parameters used by the segmentation part are marked with  "--s-..." or "-s." and parameters '
                'for the gaussian fit marked with "--g-..." or "-g.". Parameters with "CONDOR" in it are for internal Condor cluster processing.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-I', '--IN', type=str,
                    help="path to the folder 'result' to analyze. It should be used with quotes if there are spaces in the path. (str)",
                    dest="inpath")#, required=True)

## -------------------------************-----------------------
## most important gaussian parameters ; used as initial guess now -- the most import two initial parameter in Gaussain fitting. Weight_low is the highest value for a Gaussain component, sigma_high contorls the size and surface of gaussian component.
parser.add_argument('-gs', '--g-th-sigma-high', type=float,
                    help="1st important parameter: initial guess of sigma (variance) of the gaussian components in the mixture, in pixels. For Gaussian Fit. (float)",
                    dest='th_sigma_high', default=30.**0.5)
parser.add_argument('-gw', '--g-th-weight-low', type=float,
                    help="2nd important parameter: initial guess of square root of the weight of gaussian components in the mixture. For Gaussian Fit. (float)",
                    dest='th_weight_low', default=92.**0.5)#because sometime the value can be negative in the optimization, so here it's value of standard deviation.
# -------------------------*************------------------------

##other parameters
parser.add_argument('-t', '--test', dest='test', nargs='?',
                    help="If you tweak parameters and you want to test on one sequence if everything is working well. It will generate all the images for one sequence only. You can precise the site number you want to test on. Default is site 1.",
                    default=False)

parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                    help="If the debug parameter is activated, it creates the stderr and stdout files of the condor jobs.")
parser.set_defaults(debug=False)


parser.add_argument('-C', '--CONDOR', dest='CONDOR', action='store_true', help="multiple jobs in condor or not",default=False)

parser.add_argument('-maxc', '--MAXCONDOR', dest='maxthcondor', type=int, help="max in sequence for jobs of condor", default=100000)
parser.add_argument('-minc', '--MINCONDOR', dest='minthcondor', type=int, help="min in sequence for jobs of condor", default=0)

parser.add_argument('-gt', '--groundtruth', dest='compgt', action='store_true',
                    help="If you have ground truth to be compared with, you can use this parameter to compare algorithm results with ground truth. Otherwise, False.")
parser.set_defaults(compgt=True)

args = parser.parse_args(argv[1:])

args.gaussian=1#size of the gaussian kernel for smoothing each image. It should be small. For Segmentation.(int)
args.big_gaussian=2#size of the gaussian kernel for smoothing each image. It should be small. For Segmentation.(int)
args.extract_size=70#size of the core image. The rest will be used to estimate the background mean intensity level. Recommended value : 70 %% of the image size in pixels. For Segmentation. (int)
args.maxFilter=int(np.round(args.th_sigma_high*2.*1.1))#size of the region considered by the local maxima filter. It should be equal to the minimal distance between nuclei. For Segmentation. (int)


#############################################################################################

if argv[0][0] == '/':  ## not an absolute path
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

## check the input arguments
if args.inpath[-1] == '/':
    inpath = args.inpath[:-1]
else:
    inpath = args.inpath

if args.test == None:
    test_val = ' --test 1'
    args.test = '1'
elif args.test == False:
    test_val = ''
else:
    test_val = ' --test {0}'.format(args.test)


class TestBreak(Exception):
    pass


import library.sequenceExtract as sequenceExtract

temppath = inpath+'/'+inpath.split('/')[-1]+'_result'
os.chdir(temppath)

flist = next(os.walk('.'))[1]
flist.sort()
divdata = []
paramNames2 = ['Weight', 'CenterX', 'CenterY', 'rou x', 'rou y', 'sigma']

for i, ifolder in enumerate(flist):
    if ifolder.find('Pos')==-1:
        continue

    temppath2 = temppath + '/' + ifolder
    os.chdir(temppath2)
    sites_folder = glob('dir_*')
    sites_folder.sort()

    #read xls file of ground truth: if you donot have ground truth to compare, then need to comment it
    if args.compgt == True:
        with open(inpath+'/'+inpath.split('/')[-1]+'/Results'+ifolder.split('Pos')[-1].replace('_', '-')+'.xls') as f:
            data = csv.reader(f, delimiter="\t")
            d = list(data)
        numxls = len(d)-1
        dataxls = np.zeros((numxls, 3))
        for j in xrange(numxls):
            dataxls[j,1]=(float)(d[j+1][4])/1331.25*2048.0
            dataxls[j,0]=(float)(d[j+1][5])/1331.25*2048.0
            dataxls[j,2]=d[j+1][7]

    for isf, sf in enumerate(sites_folder):
        #here only for each node of condor path processing
        if isf<args.minthcondor or isf>=args.maxthcondor:
            continue

        if isf < 0:
            continue
        print('.....+++'+str(args.minthcondor)+'+++++........')
        print(isf, sf)

        para2 = []
        para3 = []
        error2 = []
        error3 = []

        os.chdir(temppath2 + '/' + sf + '/images')

        ## get the images and the sequence identification (site, x, y)
        s = sf.split('_')[0]
        x = sf.split('_')[1].split('x')[1]
        y = sf.split('_')[2].split('y')[1]
        filesSeq = glob("*.tif")
        filesSeq.sort()


        ## open and save the .gauss file
        fileout = open("{0}/{1}/seq_s{2}_x{3}_y{4}.gauss".format(temppath2, sf, s, x, y), 'w')
        fileout.write(str(args))
        fileout.write('\n')

        if (not os.path.isdir("{0}/{1}/gaussImages".format(temppath2, sf))):  #and args.test:
            os.mkdir("{0}/{1}/gaussImages".format(temppath2, sf))
        # main processing part: Gaussian fitting and division detection
        seq = sequenceExtract.Sequence(filesSeq, fileout)
        #-------------------
        # seq.studySeq(seq.treatImage, gauss = args.gaussian, gaussMask = args.big_gaussian, size = args.extract_size,
        # maxFilter = args.maxFilter, th_sigma_high = args.th_sigma_high, th_weight_low = args.th_weight_low,
        # plot = args.test, title = "{0}/gaussImages/gauss".format(inpath))
        divfr, ang,  maxind, numstd, f1, f2, f3, fgeo = seq.studySeq(seq.treatImage, gauss=args.gaussian, gaussMask=args.big_gaussian, size=args.extract_size,
                     maxFilter=args.maxFilter, th_sigma_high=args.th_sigma_high, th_weight_low=args.th_weight_low,
                     plot=False, title="{0}/{1}/gaussImages/gauss".format(temppath2, sf))

        fileout.close()
        divdata.append([sf, divfr, ang])
        # save data of division
        text_file = open(temppath + '/' + ifolder +'_divdata.txt', "a")
        if len(seq.listObj) > 0:
            text_file.write("%s,%i,%f,%f, %i, %i\n" % (sf, divfr, ang, fgeo[divfr], maxind, numstd))
        text_file.close()

        # save the data and figure results
        if len(seq.listObj) > 0:

            recnum =0
            if args.compgt == True:
                for k in xrange(len(dataxls)):
                    if int(x)-45 <= dataxls[k][0] and int(x)+45 >= dataxls[k][0] and int(y)-45 <= dataxls[k][1] and int(y)+45 >= dataxls[k][1]:
                        recnum = dataxls[k][2]
                        break
                if divfr == -1:
                    divfr = 0
                if recnum > 0:
                    recnum -= 1

            for k in range(len(seq.listObj)):
                para2.append(seq.listObj[k][0].params)
                para3.append(seq.listObj[k][1].params)
                error2.append(seq.listObj[k][0].error)
                error3.append(seq.listObj[k][1].error)
            np.savetxt(temppath2 + '/' + sf + '/para2.txt', para2, delimiter=',')
            np.savetxt(temppath2 + '/' + sf + '/para3.txt', para3, delimiter=',')
            np.savetxt(temppath2 + '/' + sf + '/error2.txt', error2, delimiter=',')
            np.savetxt(temppath2 + '/' + sf + '/error3.txt', error3, delimiter=',')
            np.save(temppath2 + '/' + sf + '/para2.npy', para2)
            np.save(temppath2 + '/' + sf + '/para3.npy', para3)
            # ---------save the feature and result figures---------------
            error2 = np.array(error2)
            error3 = np.array(error3)
            diff = (error2 - error3)/error3
            diff /= np.max(diff)
            fig1 = pylab.figure(figsize=(8, 5),dpi=100)
            pylab.subplot(2,1,1)
            pylab.plot(error2, 'r-')
            pylab.plot(error3, 'g-')
            plt.axvline(x=recnum, linewidth=2, alpha=0.5, color='c')
            plt.axvline(x=divfr,  linestyle = '--', color='k')

            pylab.subplot(2,1,2)
            pylab.plot(diff)
            plt.axvline(x=recnum, linewidth=2,  alpha=0.5, color='c')
            plt.axvline(x=divfr, linestyle = '--',  color='k')

            pylab.savefig("{0}/{1}/F1_err2and3anderrdiff_s{2}_x{3}_y{4}_t{5}.png".format(temppath2, sf, 'dir', x, y, divfr))
            pylab.close(fig1)
            #
            fig4 = pylab.figure(figsize=(35, 10),dpi=100)
            for it in xrange(12):
                pylab.subplot(2, 6 , it+1)
                pylab.plot(np.array(para2)[:, it], label="{0} in 12".format(it))
                plt.axvline(x=recnum, linewidth=2, alpha=0.5, color='c')
                plt.axvline(x=divfr, linestyle = '--',  color='k')
                pylab.title(paramNames2[it % 6])
            pylab.savefig("{0}/{1}/para2_s{2}_x{3}_y{4}.png".format(temppath2, sf, 'dir', x, y))
            pylab.close(fig4)
            #max and min distance of centers
            maxcenter_3=[]
            mincenter_3=[]
            varmin_3=[]
            center23=[]
            param23=[]
            para3=np.array(para3)
            for k in xrange(len(seq.listObj)):
                max_cdis3=np.max([np.linalg.norm(para3[k,1:3]-para3[k,7:9]),np.linalg.norm(para3[k,1:3]-para3[k,13:15]),np.linalg.norm(para3[k,7:9]-para3[k,13:15])])
                min_cdis3=np.min([np.linalg.norm(para3[k,1:3]-para3[k,7:9]),np.linalg.norm(para3[k,1:3]-para3[k,13:15]),np.linalg.norm(para3[k,7:9]-para3[k,13:15])])
                maxcenter_3.append(max_cdis3)
                mincenter_3.append(min_cdis3)
                varmin_3.append(seq.listObj[k][1].varmin)
                center23.append(seq.listObj[k][1].center23)
                param23.append(seq.listObj[k][1].param23)

            para3_b=np.copy(para3)#max and min distance of set of parameters
            for m in xrange(6):
                max_t=np.max(para3[:,m::6])
                min_t=np.min(para3[:,m::6])
                para3_b[:,m::6]=(para3_b[:,m::6]-min_t)/(max_t-min_t)
            #
            maxpara_3=[]
            minpara_3=[]
            for k in xrange(para3_b.shape[0]):
                max_cdis3=np.max([np.linalg.norm(para3_b[k,0:6]-para3_b[k,6:12]),np.linalg.norm(para3_b[k,0:6]-para3_b[k,12:18]),np.linalg.norm(para3_b[k,6:12]-para3_b[k,12:18])])
                min_cdis3=np.min([np.linalg.norm(para3_b[k,0:6]-para3_b[k,6:12]),np.linalg.norm(para3_b[k,0:6]-para3_b[k,12:18]),np.linalg.norm(para3_b[k,6:12]-para3_b[k,12:18])])
                maxpara_3.append(max_cdis3)
                minpara_3.append(min_cdis3)
            fig7 = pylab.figure(figsize=(8, 2.3),dpi=100)
            pylab.plot((varmin_3-np.min(varmin_3))/(np.max(varmin_3)-np.min(varmin_3)), 'b-')

            plt.axvline(x=recnum, linewidth=2, alpha=0.5, color='c')
            plt.axvline(x=divfr, linestyle = '--',  color='k')
            pylab.savefig("{0}/{1}/F2_var_s{2}_x{3}_y{4}_t{5}.png".format(temppath2, sf, 'dir', x, y, divfr))
            pylab.close(fig7)
            fig5 = pylab.figure(figsize=(8, 2),dpi=100)
            pylab.plot((mincenter_3-np.min(mincenter_3))/(np.max(mincenter_3)-np.min(mincenter_3)), 'g-')
            plt.axvline(x=recnum, linewidth=2, alpha=0.5, color='c')
            plt.axvline(x=divfr, linestyle = '--',  color='k')
            pylab.savefig("{0}/{1}/F3_centerdis_s{2}_x{3}_y{4}_t{5}.png".format(temppath2, sf, 'dir', x, y, divfr))
            pylab.close(fig5)

            fig8 = pylab.figure(figsize=(8, 2.3),dpi=100)
            vacenter=((mincenter_3-np.min(mincenter_3))/(np.max(mincenter_3)-np.min(mincenter_3))*diff*(varmin_3-np.min(varmin_3))/(np.max(varmin_3)-np.min(varmin_3)))**(1./3.)
            vapara=((minpara_3-np.min(minpara_3))/(np.max(minpara_3)-np.min(minpara_3))*diff*(varmin_3-np.min(varmin_3))/(np.max(varmin_3)-np.min(varmin_3)))**(1./3.)
            pylab.plot(vacenter, 'b-',label='geometric mean for centers')
            plt.axvline(x=recnum, linewidth=2.3, alpha=0.5, color='c')
            plt.axvline(x=divfr, linestyle = '--',  color='k')
            pylab.savefig("{0}/{1}/Fall_geo_s{2}_x{3}_y{4}_t{5}.png".format(temppath2, sf, 'dir', x, y, divfr))
            pylab.close(fig8)
            np.savetxt("{0}/{1}/ori_data_geo_s{2}_x{3}_y{4}.txt".format(temppath2, sf, 'dir', x, y), [error2,error3,diff,(mincenter_3-np.min(mincenter_3))/(np.max(mincenter_3)-np.min(mincenter_3)),(varmin_3-np.min(varmin_3))/(np.max(varmin_3)-np.min(varmin_3))])

            # ---gradient featues' figures---
            fig1 = pylab.figure(figsize=(8, 5),dpi=100)
            pylab.subplot(2,1,1)
            pylab.plot(error2, 'r-')#, label='fitting error of 2 components')
            pylab.plot(error3, 'g-')#, label='fitting error of 3 components')
            plt.axvline(x=recnum, linewidth=2, alpha=0.5, color='c')
            plt.axvline(x=divfr,  linestyle = '--', color='k')
            pylab.subplot(2,1,2)
            pylab.plot(f1)#,label='The difference between fitting errors of 2 and 3 components')
            plt.axvline(x=recnum, linewidth=2,  alpha=0.5, color='c')
            plt.axvline(x=divfr, linestyle = '--',  color='k')
            pylab.savefig("{0}/{1}/gra_F1_err2and3anderrdiff_s{2}_x{3}_y{4}_t{5}.png".format(temppath2, sf, 'dir', x, y, divfr))
            pylab.close(fig1)
            fig5 = pylab.figure(figsize=(8, 2),dpi=100)
            pylab.plot(f3, 'g-')
            plt.axvline(x=recnum, linewidth=2, alpha=0.5, color='c')
            plt.axvline(x=divfr, linestyle = '--',  color='k')
            pylab.savefig("{0}/{1}/gra_F3_centerdis_s{2}_x{3}_y{4}_t{5}.png".format(temppath2, sf, 'dir', x, y, divfr))
            pylab.close(fig5)
            fig7 = pylab.figure(figsize=(8, 2.3),dpi=100)
            pylab.plot(f2, 'b-')
            plt.axvline(x=recnum, linewidth=2, alpha=0.5, color='c')
            plt.axvline(x=divfr, linestyle = '--',  color='k')
            pylab.savefig("{0}/{1}/gra_F2_var_s{2}_x{3}_y{4}_t{5}.png".format(temppath2, sf, 'dir', x, y, divfr))
            pylab.close(fig7)
            fig8 = pylab.figure(figsize=(8, 2.3),dpi=100)
            pylab.plot(fgeo, 'b-',label='geometric mean for centers')
            plt.axvline(x=recnum, linewidth=2.3, alpha=0.5, color='c')
            plt.axvline(x=divfr, linestyle = '--',  color='k')
            pylab.savefig("{0}/{1}/gra_Fall_geo_s{2}_x{3}_y{4}_t{5}.png".format(temppath2, sf, 'dir', x, y, divfr))
            pylab.close(fig8)
            np.savetxt("{0}/{1}/gra_data_geo_s{2}_x{3}_y{4}.txt".format(temppath2, sf, 'dir', x, y), [f1,f2,f3,fgeo])