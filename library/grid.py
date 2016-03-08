import scipy.ndimage as ndimage
import numpy as np

import utilities 

class GridSearch(utilities.ExtractImage):
    
    def __init__(self, image, gauss = 2):
        utilities.ExtractImage.__init__(self, image, gauss = gauss)
        self.localMax = [[],[]]
        self.gridCoordinates = [[],[]]
        self.finalCoordinates = [[], []]

    def find_maxima(self, neighbor_size):
        ''' Function to find local maxima.
        find_maxima(self, neighbor_size)
        Input parameters : 
        - neighbor_size, minimum size between two local maxima ; 
        Ouput : update self.localMax'''
        maxMat = self.max_filter(neighbor_size)
        dx, dy = np.indices(self.shape)
        y = dx[maxMat == 1]
        x = dy[maxMat == 1]
        self.localMax = [list(x),list(y)]

    def find_grid(self, neighbor_size):
        '''Function to find the grid coordinates. For now this function is only for a grid parallel to the edges.
        find_grid(self, neighbor_size)
        Input : 
        - neighbor_size, minimal distance between two lines or columns in the grid (assumption : symetric grid) ;
        Output : update of self.gridCoordinates '''
        y = np.sum(self.image, axis = 1)
        x = np.sum(self.image, axis = 0)
        x_filter = ndimage.filters.maximum_filter(x, size = neighbor_size)
        y_filter = ndimage.filters.maximum_filter(y, size = neighbor_size)
        x_i = np.arange(len(x))[x == x_filter]
        y_i = np.arange(len(y))[y == y_filter]
        x_out, y_out = [], []
        for i in x_i:
            for j in y_i:
                x_out.append(i)
                y_out.append(j)
        self.gridCoordinates = [x_out, y_out]
        
    def exclude_edging(self, frame_size, final = True):
        ''' Function to remove points within frame_size/2 of image edges.
        exclude_edging(self, frame_size, localOrGrid)
        Input parameters :
        - frame_size, twice the excluding size from edges ;
        - fianl, boolean, if True finalCoordinates (barycenter of the coupel max+grid), if False max. Maxima to update.
        Output : update of the choosen maxima. '''
        to_remove = []
        if final:
            x, y = self.finalCoordinates
        else:
            x, y = self.localMax
        for i in xrange(len(x)):
            if x[i] > self.shape[0] - frame_size/2. or x[i] < frame_size/2. or \
            y[i] > self.shape[0] - frame_size/2. or y[i] < frame_size/2.:
                to_remove.append(i)
        for count, i in enumerate(to_remove):
            x.pop(i-count)
            y.pop(i-count)
        if final:
            self.finalCoordinates = [x,y]
        else:
            self.localMax = [x,y]
        return x, y
       
    def euclidianbwMax(self):
        x1,y1 = self.localMax
        x2, y2 = self.gridCoordinates
        diff_x1 = np.tile(x1,len(x2))
        diff_x1 = np.reshape(diff_x1, (len(x2), len(x1)))
        diff_x2 = np.repeat(x2,len(x1))
        diff_x2 = np.reshape(diff_x2, (len(x2), len(x1)))
        diff_X = diff_x2 - diff_x1
        diff_y1 = np.tile(y1,len(y2))
        diff_y1 = np.reshape(diff_y1, (len(y2), len(y1)))
        diff_y2 = np.repeat(y2,len(y1))
        diff_y2 = np.reshape(diff_y2, (len(y2), len(y1)))
        diff_Y = diff_y2 - diff_y1
        return np.sqrt(diff_X*diff_X+diff_Y*diff_Y)
    
    def match_max(self, distmax):    
        ''' Function which returns the grid maxima corresponding to local maxima, ie at the maximum distance of 'distmax' of a local maximum.
        match_max(self, distmax)
        Input : distmax, (int).
        Output : two lists x and y of the matching maxima (coordinates are calculted as barycenter of the couple max + grid).'''
        distMat = self.euclidianbwMax()
        xmax, ymax = self.localMax
        gridx, gridy = self.gridCoordinates
        map_maxTOgrid = [-1]*len(xmax)
        length = len(distMat[0,:])
        min_ = np.min(distMat)
        while min_ < distmax :
            i = np.argmin(distMat)/length
            j = np.argmin(distMat)%length
            if map_maxTOgrid[j] == -1:   
                map_maxTOgrid[j] = i
            distMat[i,j] = distmax
            min_ = np.min(distMat)
        xout, yout = [], []
        for j,i in enumerate(map_maxTOgrid):
            if i >= 0:
                xout.append((gridx[i]+xmax[j])/2)
                yout.append((gridy[i]+ymax[j])/2)
        self.finalCoordinates = [xout, yout]
        
        