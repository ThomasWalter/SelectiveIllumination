import os, re, time, sys
import shutil
import types

import vigra
#import vigra.morpho

from settings import *

import numpy as np

from scipy.stats import nanmean


class Utilities(object):

    def overlay_segmentation_result(self, img, bin_img, filename, color=[255, 0, 0], 
                                    contour=False):
        if len(img.shape) == 2:
            width, height = img.shape
            nb_channels = 1
        elif len(img.shape) == 3:
            width, height, d = img.shape
            nb_channels = d
        else: 
            print 'Image shape not supported.'
            return 
        rgb_img = vigra.RGBImage((width, height))
#        red, green, blue = rgb_img.channelIter()
#        red = img.copy()
#        green = img.copy()
#        blue = img.copy()
#        red[bin_img.reshape((width, height)) > 0] = color[0]
#        green[bin_img.reshape((width, height)) > 0] = color[1]
#        blue[bin_img.reshape((width, height)) > 0] = color[2]

        
        for i, channel in enumerate(rgb_img.channelIter()):
            if nb_channels == 1:
                # if the original image is a grey level image, we copy the image to each channel.
                channel.copyValues(img)
            else:
                if i < d:
                    # if the image is already a color image, then we copy the channel
                    channel.copyValues(img[:,:,i])
                                    
            if contour:
                se = vigra.morpho.structuringElement2D([[-1, -1], [0, -1] , [1, -1],
                                                        [-1,  0], [0,  0] , [1,  0],
                                                        [-1,  1], [0,  1] , [1,  1],
                                                        ])
                se.size=1
                grad_img = vigra.morpho.morphoInternalGradient(bin_img, se)
                channel[grad_img.reshape((width, height)) > 0] = color[i]
            else:
                channel[bin_img.reshape((width, height)) > 0] = color[i]
            
        vigra.impex.writeImage(rgb_img, filename)
                
        return

    def overlay_segmentation_result_list(self, img, bin_img, filename, colors=None, 
                                         contour=False):
        if len(img.shape) == 2:
            width, height = img.shape
            nb_channels = 1
        elif len(img.shape) == 3:
            width, height, d = img.shape
            nb_channels = d
        else: 
            print 'Image shape not supported.'
            return 
            
        if colors is None:
            cm = plotter.colors.ColorMap()
            if type(bin_img) == types.ListType: 
                N = length(bin_img)
            else:
                N = bin_img.shape[-1]
            if N == 1: 
                colors = [(255.0, 0.0, 0.0)]
            else:
                colors = [[255*y for y in x] for x in cm.makeDivergentColorRamp(N)]
        else:            
            N = len(colors)
            
        rgb_img = vigra.RGBImage((width, height))
#        red, green, blue = rgb_img.channelIter()
#        red = img.copy()
#        green = img.copy()
#        blue = img.copy()
#        red[bin_img.reshape((width, height)) > 0] = color[0]
#        green[bin_img.reshape((width, height)) > 0] = color[1]
#        blue[bin_img.reshape((width, height)) > 0] = color[2]

        
        for i, channel in enumerate(rgb_img.channelIter()):
            if nb_channels == 1:
                # if the original image is a grey level image, we copy the image to each channel.
                channel.copyValues(img)
            else:
                if i < d:
                    # if the image is already a color image, then we copy the channel
                    channel.copyValues(img[:,:,i])
            
            for k in range(N):
                if type(bin_img) == types.ListType: 
                    bi = bin_img[k]
                else:
                    bi = bin_img[:,:,k]

                color = colors[k]
                
                if contour:
                    se = vigra.morpho.structuringElement2D([[-1, -1], [0, -1] , [1, -1],
                                                            [-1,  0], [0,  0] , [1,  0],
                                                            [-1,  1], [0,  1] , [1,  1],
                                                            ])
                    se.size=1
                    grad_img = vigra.morpho.morphoInternalGradient(bi, se)
                    channel[grad_img.reshape((width, height)) > 0] = color[i]
                else:
                    channel[bi.reshape((width, height)) > 0] = color[i]
            
        vigra.impex.writeImage(rgb_img, filename)
                
        return
        
    def export_image(self, img, filename):
        vigra.impex.writeImage(self.to_uint8(img), filename)
        return
        
    def to_uint8(self, img):
        return img.astype(np.dtype('uint8'))

    def to_uint16(self, img):
        return img.astype(np.dtype('uint16'))

    def to_uint32(self, img):
        return img.astype(np.dtype('uint32'))
    
    def to_float(self, img):
        return img.astype(np.dtype('float32'))
    
    def make_vigra_image(self, X):
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))
        elif len(X.shape) < 2 or len(X.shape) > 3:
            print 'shape of array: ', X.shape
            print 'conversion was not successful.'
            return None        
        img = vigra.taggedView(X, vigra.defaultAxistags('xyc'))
        return img
    
