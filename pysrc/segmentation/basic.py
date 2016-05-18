import os, re, time, sys
import shutil
import types

# skimage imports
import skimage
import skimage.io
from skimage.morphology import disk
from skimage import morphology
from skimage.filters import rank
from skimage import filters
from skimage import color
from skimage import restoration
from skimage.measure import label
from skimage import measure
from skimage.morphology import watershed
from skimage.feature import peak_local_max

from scipy import ndimage as ndi

from settings import *

import numpy as np

from optparse import OptionParser

from Queue import Queue

import pdb

import skimage.draw
from lxml.html.builder import IMG


class Overlay(object):
    def to_gray_scale(self, img, imbin, colorvalue=(255, 0, 0), alpha=1.0,
                      gradient=True):
        max_color = max(colorvalue)
        if max_color > 1.0:
            colorvalue = [a / max_color for a in colorvalue]
        colim = color.gray2rgb(img)
        
        if gradient:
            #se = morphology.disk(1)
            se = morphology.rectangle(3,3)
            imvis = imbin - morphology.erosion(imbin, se)
        else:
            imvis = imbin
        for i, col in enumerate(colorvalue):
            channel_img = colim[:,:,i]
            
            channel_img[imvis>0] = (1-alpha) * channel_img[imvis>0]
            
            colim[:,:,i] = alpha*col*imvis + channel_img
        
        return colim
    
class SimpleWorkflow(object):
    
    def __init__(self, settings_filename=None, settings=None, prefix=''):
        if settings is None and settings_filename is None:
            raise ValueError("Either a settings object or a settings filename has to be given.")
        if not settings is None:
            self.settings = settings
        elif not settings_filename is None:
            self.settings = Settings(os.path.abspath(settings_filename), dctGlobals=globals())
        
        print self.settings
        
        for folder in self.settings.make_folder:
            if not os.path.isdir(folder):
                print 'made folder %s' % folder
                os.makedirs(folder)

        self.ov = Overlay()
        self.prefix=prefix
                              
    def __call__(self, img_16bit):
        
        print '16 bit --> 8 bit'
        img = self.reduce_range(img_16bit, minmax=True)
        
        print 'prefiltering'
        pref = self.prefilter(img, self.settings.segmentation_settings['prefiltering'])
        
        if self.settings.debug:
                
            out_filename = os.path.join(self.settings.img_debug_folder, '%s01_original.png' % self.prefix )
            skimage.io.imsave(out_filename, img)
            print 'wrote: ', out_filename
            
            out_filename = os.path.join(self.settings.img_debug_folder, '%s02_prefiltered.png' % self.prefix) 
            skimage.io.imsave(out_filename, pref)
            print 'wrote: ', out_filename
            
        
        bgsub = self.background_subtraction(pref, self.settings.segmentation_settings['bg_sub'])
        hmax = self.homogenize(pref)
        
        if self.settings.debug:
            out_filename = os.path.join(self.settings.img_debug_folder, '%s03_bgsub.png' % self.prefix) 
            skimage.io.imsave(out_filename, bgsub)

            out_filename = os.path.join(self.settings.img_debug_folder, '%s04_hmax.png' % self.prefix) 
            skimage.io.imsave(out_filename, hmax)

        
        # vigra 
        # width, height, d = img.shape
        # segmentation_result = vigra.Image((width, height))
        
        # skimage
        segmentation_result = np.zeros(img.shape)
        thresh = self.settings.segmentation_settings['thresh']
        segmentation_result[bgsub>thresh] = 255
        segmentation_result = segmentation_result.astype(img.dtype)

        seg_hmax = np.zeros(img.shape)
        thresh = self.settings.segmentation_settings['thresh_hmax']
        seg_hmax[hmax>thresh] = 255
        seg_hmax = seg_hmax.astype(img.dtype)

        if self.settings.debug:
            out_filename = os.path.join(self.settings.img_debug_folder, '%s05_local_adaptive_threshold.png' % self.prefix)  
            skimage.io.imsave(out_filename, segmentation_result)

            out_filename = os.path.join(self.settings.img_debug_folder, '%s06_seghmax.png' % self.prefix)
            skimage.io.imsave(out_filename, seg_hmax)

        # final segmentation (before split)
        segmentation_result[seg_hmax>0] = 255

        if self.settings.debug:            
            out_filename = os.path.join(self.settings.img_debug_folder, '%s07_segres.png' % self.prefix)
            skimage.io.imsave(out_filename, segmentation_result)

        # hole filling: 
        temp = segmentation_result.copy()
        se = disk(self.settings.segmentation_settings['hole_size'])
        dil = morphology.dilation(temp, se)
        segmentation_result = morphology.reconstruction(dil, temp, method='erosion')
        segmentation_result = segmentation_result.astype(temp.dtype)
        
        if self.settings.debug:            
            out_filename = os.path.join(self.settings.img_debug_folder, '%s07_segres_holes_filled.png' % self.prefix)
            skimage.io.imsave(out_filename, segmentation_result)

            overlay_img = self.ov.to_gray_scale(img, segmentation_result, (1.0, 0.0, 0.0), 0.8)
            out_filename = os.path.join(self.settings.img_debug_folder, '%s08_overlay_both.png' % self.prefix)
            skimage.io.imsave(out_filename, overlay_img)
                    
        # split
        labres = self.split_cells(segmentation_result, img)
        wsl = self.filter_wsl(segmentation_result, labres, img)
        #wsl = self.get_internal_wsl(labres)
        res = segmentation_result.copy()
        res[wsl>0] = 0

        if self.settings.debug:
            out_filename = os.path.join(self.settings.img_debug_folder, '%s09_after_split.png' % self.prefix)
            skimage.io.imsave(out_filename, res)

            overlay_img = self.ov.to_gray_scale(img, res, (1.0, 0.0, 0.0), 0.8)
            out_filename = os.path.join(self.settings.img_debug_folder, '%s10_overlay_after_split.png' % self.prefix)
            skimage.io.imsave(out_filename, overlay_img)
        
        # postfiltering
        res = self.postfilter(res, img)

        if self.settings.debug:
            out_filename = os.path.join(self.settings.img_debug_folder, '%s11_postfilter.png' % self.prefix)
            skimage.io.imsave(out_filename, res)

            overlay_img = self.ov.to_gray_scale(img, res, (1.0, 0.0, 0.0), 0.8)
            out_filename = os.path.join(self.settings.img_debug_folder, '%s12_overlay_postfilter.png' % self.prefix)
            skimage.io.imsave(out_filename, overlay_img)

            final_label = label(res, neighbors=4, background=0)
            final_label = final_label - final_label.min()
            out_filename = os.path.join(self.settings.img_debug_folder, '%s12bis_labels_after_postfilter.png' % self.prefix)
            skimage.io.imsave(out_filename, final_label)

        res = self.remove_border_objects(res)

        if self.settings.debug:
            out_filename = os.path.join(self.settings.img_debug_folder, '%s13_border_obj_removed.png' % self.prefix)
            skimage.io.imsave(out_filename, res)

            overlay_img = self.ov.to_gray_scale(img, res, (1.0, 0.0, 0.0), 0.8)
            out_filename = os.path.join(self.settings.img_debug_folder, '%s14_overlay_border_obj_removed.png' % self.prefix)                                                                                
            skimage.io.imsave(out_filename, overlay_img)

            final_label = label(res, neighbors=4, background=0)
            final_label = final_label - final_label.min()
            out_filename = os.path.join(self.settings.img_debug_folder, '%s15_final_label.png' % self.prefix)
            skimage.io.imsave(out_filename, final_label)
            
        return res
    
    def get_large_wsl(self, labelimage):
        se = morphology.diamond(1)
        dil = morphology.dilation(labelimage, se)
        ero = morphology.erosion(labelimage, se)
        grad = dil - ero
        res = np.zeros(labelimage.shape)
        res[grad>0] = 255
        return res

    def get_internal_wsl(self, labelimage):
        se = morphology.diamond(1)
        ero = morphology.erosion(labelimage, se)
        grad = labelimage - ero
        res = np.zeros(labelimage.shape)
        res[grad>0] = 255
        return res

    def get_external_wsl(self, labelimage):
        #se = morphology.square(3)
        se = morphology.diamond(1)
        dil = morphology.dilation(labelimage, se)
        grad = dil - labelimage
        res = np.zeros(labelimage.shape)
        res[grad>0] = 255
        return res
    
    def remove_border_objects(self, imbin):
        labelim = label(imbin, neighbors=4, background=0)
        labelim = labelim - labelim.min()
        
        A = np.hstack([labelim[0,:], labelim[-1,:], labelim[:,0], labelim[:,-1]])
        border_counts = np.bincount(A)
        if len(border_counts) < labelim.max() + 1:
            delta = labelim.max() + 1 - len(border_counts)
            border_counts = np.hstack([border_counts, np.zeros(delta)])            
        
        border_filter = np.where(border_counts > 0, 255, 0)
        
        to_remove = border_filter[labelim]   
        imbin[to_remove>0] = 0        
        
        return imbin
    
    # arguments : segmentation_result, labres, img
    def filter_wsl(self, imbin, ws_labels, imin):
        
        # internal gradient of the cells: 
        se = morphology.diamond(1)
        #ero = morphology.erosion(imbin, se)        
        #grad = imbin - ero
        
        # watershed line        
        wsl = self.get_external_wsl(ws_labels)
        #wsl = self.get_large_wsl(ws_labels)
        wsl_remove = wsl.copy()

        # watershed line outside the cells is 0
        wsl_remove[imbin==0] = 0
        # watershed line on the gradient (border of objects)
        # is also not considered
        #wsl_remove[grad>0] = 0
                
        # gradient image
        pref = 255 * filters.gaussian_filter(imin, 3.0)
        pref[pref < 0] = 0
        pref = pref.astype(np.dtype('uint8'))
        ero = morphology.erosion(pref, se)
        dil = morphology.dilation(pref, se)
        grad = dil - ero
        grad_filtered = grad
        
        if self.settings.debug:
            out_filename = os.path.join(self.settings.img_debug_folder, '%s09_watershed_regions.png' % self.prefix)
            skimage.io.imsave(out_filename, ws_labels.astype(np.dtype('uint8')))        

            out_filename = os.path.join(self.settings.img_debug_folder, '%s09_wsl.png' % self.prefix)
            skimage.io.imsave(out_filename, wsl.astype(np.dtype('uint8')))        

            out_filename = os.path.join(self.settings.img_debug_folder, '%s09_wsl_remove.png' % self.prefix)
            skimage.io.imsave(out_filename, wsl_remove.astype(np.dtype('uint8')))        

            out_filename = os.path.join(self.settings.img_debug_folder, '%s09_wsl_gradient.png' % self.prefix)            
            skimage.io.imsave(out_filename, grad_filtered.astype(np.dtype('uint8')))        
        
        labimage = label(wsl_remove)
        properties = measure.regionprops(labimage, grad_filtered)   
        
        mean_intensities = np.array([0.0] + [pr.mean_intensity for pr in properties])
        filter_intensities = np.where(mean_intensities < self.settings.postfilter['wsl_mean_intensity'], 255, 0)
        filter_intensities[0] = 0
        
        wsl_remove = filter_intensities[labimage]
        #print filter_intensities
        #print mean_intensities
        wsl[wsl_remove>0] = 0

        if self.settings.debug:
            out_filename = os.path.join(self.settings.img_debug_folder, '%s09_wsl_remove2.png' % self.prefix)
            skimage.io.imsave(out_filename, wsl_remove.astype(np.dtype('uint8')))        

        return wsl
    
    def split_cells(self, segres, imin):
        ssc = self.settings.split_cells
        
        distance = ndi.distance_transform_edt(segres)
        
        if ssc['sigma'] > 0 : 
            distance_filtered = filters.gaussian_filter(distance, ssc['sigma'])
        else: 
            distance_filtered = distance

        local_maxima = self.local_maxima(distance_filtered, ssc['h'])
        #local_maxima = peak_local_max(distance_filtered, indices=False, 
        #                              footprint=np.ones((3, 3)), 
        #                              min_distance=20)
        #local_maxima[local_maxima>0] = 255
        
        if self.settings.debug:
            out_filename = os.path.join(self.settings.img_debug_folder, '%s09_distance_function.png' % self.prefix)
            skimage.io.imsave(out_filename, distance.astype(np.dtype('uint8')))

            out_filename = os.path.join(self.settings.img_debug_folder, '%s09b_distance_filtered.png' % self.prefix)
            skimage.io.imsave(out_filename, distance_filtered.astype(np.dtype('uint8')))

            out_filename = os.path.join(self.settings.img_debug_folder, '%s10_local_maxima.png' % self.prefix)
            lm = local_maxima.astype(np.dtype('uint8'))
            lm[lm>0] = 255
            skimage.io.imsave(out_filename, lm)
            

        #local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
        #                            labels=image)
        dist_img = distance_filtered.max() - distance_filtered
        dist_img = dist_img.astype(np.dtype('int64'))
        markers = label(local_maxima)
        labels = watershed(dist_img, markers, mask=segres)
        
        return labels
    
    def postfilter(self, segres, img):
        labimage = label(segres, connectivity=1) # we use 4-connectivity
        properties = measure.regionprops(labimage, img)
        
        areas = np.array([0] + [pr.area for pr in properties])
        filter_areas = np.where(areas > self.settings.postfilter['area'], 255, 0)
        #res = filter_areas[labimage]
        
        mean_intensities = np.array([0.0] + [pr.mean_intensity for pr in properties])
        filter_intensities = np.where(mean_intensities > self.settings.postfilter['mean_intensity'], 255, 0)

        total_filter = np.min(np.vstack([filter_areas, filter_intensities]), axis=0)
        res = total_filter[labimage]
        
        return res
        
    def reduce_range(self, img, minmax=False):
        
        minval = np.min(img)
        maxval = np.max(img)
        if minmax:
            delta = maxval - minval
            if delta <= 0: delta = 1.0
            temp = 255.0 / delta * (img - minval)            
        else:
            temp = 255.0 / 2**16 * (img - minval)
        res = temp.astype(np.uint8)
        return res
    
        # if filename is given, the image is written. filename should be the full path.     
    def background_subtraction(self, img, method='avg'):
        #width, height = img.shape
        
        if method=='avg': 
            # vigra
            #kernel = vigra.filters.averagingKernel(radius)
            #bgsub = img - vigra.filters.convolve(self.ut.to_float(img), kernel)
            
            # with skimage           
            se = disk(self.settings.background_subtraction['radius'])
            bgsub = img.astype(np.dtype('float')) - rank.mean(img, se)
            bgsub[bgsub < 0] = 0
            bgsub = bgsub.astype(img.dtype)

        if method=='med': 
            # vigra
            #kernel = vigra.filters.averagingKernel(radius)
            #bgsub = img - vigra.filters.convolve(self.ut.to_float(img), kernel)
            
            # with skimage           
            se = disk(self.settings.background_subtraction['radius'])
            bgsub = img.astype(np.dtype('float')) - rank.median(img, se)
            bgsub[bgsub < 0] = 0
            bgsub = bgsub.astype(img.dtype)
            
        elif method=='constant_median':
            # vigra
            #bgsub = img - np.median(np.array(img))
            
            # with skimage            
            bgsub = img - np.median(img)

            bgsub[bgsub < 0] = 0
            bgsub = bgsub.astype(img.dtype)
            
        return bgsub
    
    
    def prefilter_test(self, img_16bit):
        img = self.reduce_range(img_16bit)
        
        filename = os.path.join(self.settings.debug_prefilter_test, 'img_0_original.png')
        skimage.io.imsave(filename, img)
            
        for k, method in enumerate(['median', 'close_rec',
                                    'avg', 'denoise_bilateral', 'denbi_clorec', 
                                    'denbi_asfrec']):
            
            start_time = time.time()
            pref = self.prefilter(img, method)
            stop_time_pref = time.time()
            diff_time = stop_time_pref - start_time
            homo = self.homogenize(pref)
            diff_time_homo = time.time() - stop_time_pref
                        
            filename = os.path.join(self.settings.debug_prefilter_test, 'img_%i_%s.png' % (k+1, method))
            skimage.io.imsave(filename, pref)
            print 'result written to file: %s' % filename
            
            filename = os.path.join(self.settings.debug_prefilter_test, 'homo_%i_%s.png' % (k+1, method))
            maxval = np.max(homo)
            minval = np.min(homo)
            if maxval > minval: 
                k = 255.0 / (maxval - minval)
            else:
                k = 1.0
            homo_export = k*(homo.astype(np.dtype('float')) - minval)
            homo_export = homo_export.astype(np.dtype('uint8'))            
            skimage.io.imsave(filename, homo_export)
            print 'homo result written to file: %s' % filename
            
            print 'range: %f %f %s' % (np.min(pref), np.max(pref), pref.dtype)
            print '%s\telapsed time: %02i:%02i:%03i' % (method, (diff_time / 60), (diff_time % 60), int(diff_time_homo % 1 * 1000)) 
            print '%s\telapsed time (homo): %02i:%02i:%03i' % (method, (diff_time_homo / 60), (diff_time_homo % 60), int(diff_time_homo % 1 * 1000))            

        return
    
    def local_maxima(self, img, h=1):
        img_sub = img.astype(np.dtype('float')) - h
        img_sub[img_sub<0] = 0
        img_sub = img_sub.astype(img.dtype)
        
        # seed and then mask
        temp = morphology.reconstruction(img_sub, img)
        res = img - temp.astype(img.dtype)
        res[res>0] = 255        
        res = res.astype(np.dtype('uint8'))

        return res    
    
    def homogenize(self, img):
        img_sub = img.astype(np.dtype('float')) - self.settings.homogenize_settings['h']
        img_sub[img_sub<0] = 0
        img_sub = img_sub.astype(img.dtype)
        
        # seed and then mask
        print 'img: ', np.min(img), np.max(img)
        print 'img_sub: ', np.min(img_sub), np.max(img_sub)
        temp = morphology.reconstruction(img_sub, img)
        res = img - temp.astype(img.dtype)
        return res
    
    def prefilter(self, img, method='median'):
        
        ps = self.settings.prefilter_settings[method]
        
        print
        print 'prefiltering :', method

        if method=='median':
            radius = ps['median_size']

            # with vigra
            #filtered= vigra.filters.discMedian(img, radius)
            
            # with skimage            
            pref = rank.median(img, disk(radius))
            
        elif method=='avg': 
            # with skimage           
            se = disk(ps['avg_size'])   
            pref = rank.mean(img, se)         
            
        elif method=='bilateral': 
            # with skimage           
            se = disk(ps['bil_radius']) 
            pref = rank.mean_bilateral(img, se,
                                       s0=ps['bil_lower'], 
                                       s1=ps['bil_upper'])
        
        elif method=='denoise_bilateral':
            #skimage.filters.denoise_bilateral(image, win_size=5, sigma_range=None, sigma_spatial=1,

            pref = restoration.denoise_bilateral(img, ps['win_size'], ps['sigma_signal'], ps['sigma_space'], ps['bins'], 
                                                 mode='constant', cval=0, multichannel=False)
                       
        elif method=='close_rec':
            se = disk(ps['close_size'])
            dil = morphology.dilation(img, se)
            rec = morphology.reconstruction(dil, img, method='erosion')
            
            # reconstruction gives back a float image (for whatever reason). 
            pref = rec.astype(dil.dtype)
            
        elif method=='denbi_clorec':
            temp = restoration.denoise_bilateral(img, ps['win_size'], ps['sigma_signal'], ps['sigma_space'], ps['bins'], 
                                                 mode='constant', cval=0, multichannel=False)
            temp = 255 * temp
            temp = temp.astype(img.dtype)
            
            se = disk(ps['close_size'])
            dil = morphology.dilation(temp, se)
            rec = morphology.reconstruction(dil, temp, method='erosion')
            
            # reconstruction gives back a float image (for whatever reason). 
            pref = rec.astype(img.dtype)            

        elif method=='denbi_asfrec':
            temp = restoration.denoise_bilateral(img, ps['win_size'], ps['sigma_signal'], ps['sigma_space'], ps['bins'], 
                                                 mode='constant', cval=0, multichannel=False)
            temp = 255 * temp
            temp = temp.astype(img.dtype)
            
            se = disk(ps['close_size'])
            dil = morphology.dilation(temp, se)
            rec = morphology.reconstruction(dil, temp, method='erosion')

            se = disk(ps['open_size'])
            ero = morphology.erosion(rec, se)
            rec2 = morphology.reconstruction(ero, rec, method='dilation')
            
            # reconstruction gives back a float image (for whatever reason). 
            pref = rec2.astype(img.dtype)            

        elif method=='med_denbi_asfrec':
            if ps['median_size'] > 1:
                radius = ps['median_size']
                pref = rank.median(img, disk(radius))
            else:
                pref = img
 
            temp = restoration.denoise_bilateral(pref, ps['win_size'], ps['sigma_signal'], ps['sigma_space'], ps['bins'], 
                                                 mode='constant', cval=0, multichannel=False)
            temp = 255 * temp
            temp = temp.astype(img.dtype)
            
            if ps['close_size'] > 0 : 
                se = disk(ps['close_size'])
                dil = morphology.dilation(temp, se)
                rec = morphology.reconstruction(dil, temp, method='erosion')
            else:
                rec = temp
                
            if ps['open_size'] > 0:
                se = disk(ps['open_size'])
                ero = morphology.erosion(rec, se)
                rec2 = morphology.reconstruction(ero, rec, method='dilation')
            else:
                rec2 = rec
            
            # reconstruction gives back a float image (for whatever reason). 
            pref = rec2.astype(img.dtype)            
            
        
        return pref
    

class BatchProcessor(object):
    
    def __init__(self, settings_filename=None, settings=None):
        if settings is None and settings_filename is None:
            raise ValueError("Either a settings object or a settings filename has to be given.")
        if not settings is None:
            self.settings = settings
        elif not settings_filename is None:
            self.settings = Settings(os.path.abspath(settings_filename), dctGlobals=globals())
        
        for folder in self.settings.make_folder:
            if not os.path.isdir(folder):
                print 'made folder %s' % folder
                os.makedirs(folder)

        self.segmenter = SimpleWorkflow(settings=self.settings)
        #self.ut = utilities.Utilities()
        
    
    def __call__(self, filenames=None):
        if filenames is None:
            filenames = filter(lambda x: os.path.splitext(x)[-1] in ['.tif', '.tiff', '.TIFF', '.TIF'], 
                               os.listdir(self.settings.data_folder))

        for filename in filenames:
            
            # import with vigra:
            #temp_img = vigra.readImage(os.path.join(self.settings.data_folder, filename))
            #img = self.ut.to_uint8(temp_img)

            # import with skimage
            temp_img = skimage.io.imread(os.path.join(self.settings.data_folder, filename))                        
            if len(temp_img.shape) > 2 :
                img = temp_img[1,:,:]
            else:
                img = temp_img
            
            res = self.segmenter(img)
            
            out_filename = os.path.join(self.settings.segmentation_folder,
                                        'segmentation_%s' % filename.replace(' ', '_'))
            print 'writing %s' % out_filename
                        
            # export with vigra
            #vigra.impex.writeImage(res, out_filename)

            # export with skimage
            skimage.io.imsave(out_filename, res.astype('uint8'))
            
        return
    
    
def test_script():
    filename = '/Users/twalter/data/Perrine/max_proj/C1-PM84 2.12.14 MDCK SnailERT2 H2B Dendra2 live med sort photoconvo 1.tif'

    import skimage
    import skimage.io
    import segmentation.basic
    
    temp_img = skimage.io.imread(filename)
    img = temp_img[1,:,:]
    sw = segmentation.basic.SimpleWorkflow("./project_settings/settings_2015_09_18.py")
    imin = sw.reduce_range(img)
    bp = segmentation.basic.BatchProcessor("./project_settings/settings_2015_09_18.py")
    res = bp.segmenter(img)
    sel = segmentation.basic.Select("./project_settings/settings_2015_09_18.py")
    sel(imin, res, 1)
    return

if __name__ ==  "__main__":

    description =\
'''
%prog - running segmentation tool .
'''

    parser = OptionParser(usage="usage: %prog [options]",
                         description=description)

    parser.add_option("-s", "--settings_file", dest="settings_file",
                      help="Filename of the settings file")

    (options, args) = parser.parse_args()

    if (options.settings_file is None):
        parser.error("incorrect number of arguments!")

    bp = BatchProcessor(options.settings_file)
    bp()
    
    
        