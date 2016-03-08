# livespin
Python implementation of the methods decribed in : "Detection and tracking of overlapping cell nuclei for large scale mitosis analyses" by Yingbo Li, France Rose, Florencia di Pietro, Xavier Morin, and Auguste Genovesio

The code was written by Yingbo Li and France Rose under supervision of Auguste Genovesio.

# Installation guide:
It is strongly recommended to install Anaconda for Python 2.7 because most of the required libraries are included in this package.
The remaining needed libraries can be installed through PIP this way after you install Anaconda:
  1. pip install tifffile
  2. pip install tiffcapture
(If you run into installation errors using pip, you may have to install a C++ Compiler for Python 2.7 (e.g. from http://aka.ms/vcpython27)

OpenCV is also required. 

Then you can download the code by clicking the ZIP link on this webpage or by cloning the whole project using: git clone https://github.com/lantuzi/livespin

# Description:
  1. The library folder contains the functions used by the two main scripts below.
  2. a_extract_cy5.py extract and crop image sequences taken on each micropattern from a larger field of view.
  3. b_detectandident.py identify the timeframe, the location and the angle of the second cell division in a sequence. This script can be used on a stand alone computer but provides all necessary options to be run in parallel on a computing cluster using the HTCondor scheduller (https://research.cs.wisc.edu/htcondor/).
  





