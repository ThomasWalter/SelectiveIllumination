import os, re, time, sys
import shutil
import types

#import vigra
#import vigra.morpho

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

#from utilities import Utilities
from misc import utilities

import pdb
from test.test_support import temp_cwd
from rdflib.plugins.parsers.pyRdfa.transform.prototype import pref

class Select(object):
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

        self.ov = Overlay()
        self.sw = SimpleWorkflow(settings=self.settings)
        
    def find_neighbors(self, imbin, max_extension):
        background = np.zeros(imbin.shape)
        background[imbin==0] = 255
        distance = ndi.distance_transform_edt(background)
        cell_labels = label(imbin, neighbors=4)

        mask = np.zeros(imbin.shape)
        mask[distance < max_extension] = 255
        
        labels = watershed(distance, cell_labels, mask=mask)

        if self.settings.debug:
            out_filename = os.path.join(self.settings.cell_selection_folder, 'distance.png')
            temp = distance / distance.max()
            skimage.io.imsave(out_filename, temp)

            out_filename = os.path.join(self.settings.cell_selection_folder, 'labels_from_ws.png')
            skimage.io.imsave(out_filename, labels)
        
        
        return labels
    
    def get_properties(self, imbin, img):
        props = {}
        cell_labels = label(imbin, neighbors=4)
        properties = measure.regionprops(cell_labels, img)
        areas = [0] + [pr.area for pr in properties]
        mean_intensities = [0.0] + [pr.mean_intensity for pr in properties]        
        eccentricities = [0.0] + [pr.eccentricity for pr in properties]
        for i in range(1, cell_labels.max()+1):
            props[i] = {
                        'area': areas[i],
                        'mean_intensity': mean_intensities[i],
                        'eccentricity': eccentricities[i],
                        }
        return props
    
    def distance_to_avg(self, props):

        # calc average: 
        mean_area = np.mean([props[i]['area'] for i in props.keys()])
        std_area = np.std([props[i]['area'] for i in props.keys()])
        mean_intensity = np.mean([props[i]['mean_intensity'] for i in props.keys()])
        std_intensity = np.std([props[i]['mean_intensity'] for i in props.keys()])
        mean_ecc = np.mean([props[i]['eccentricity'] for i in props.keys()])
        std_ecc = np.std([props[i]['eccentricity'] for i in props.keys()])
        
        if std_area==0.0: std_area = 1.0
        if std_intensity==0.0: std_intensity = 1.0
        if std_ecc==0.0: std_ecc = 1.0
            
        for i in props.keys(): 
#             temp = (props[i]['area'] - mean_area)**2 + \
#                    (props[i]['mean_intensity'] - mean_intensity)**2 + \
#                    (props[i]['eccentricity'] - mean_ecc)**2
            temp = ((props[i]['area'] - mean_area) / std_area)**2 + \
                   ((props[i]['mean_intensity'] - mean_intensity) / std_intensity)**2 + \
                   ((props[i]['eccentricity'] - mean_ecc) / std_ecc)**2

            props[i]['distance'] = np.sqrt(temp)
                
        return props
    
    def select_label(self, labels=None):
        
        if labels is None:
            keys = sorted(filter(lambda x: self.nodes[x]['allowed'], self.nodes.keys()))
        elif len(labels) > 0:
            keys = sorted(filter(lambda x: self.nodes[x]['allowed'], labels))
        else:
            keys = []
        if len(keys) == 0:
            return None
        
        distances = [self.nodes[i]['distance'] for i in keys]        
        min_distance = np.min(distances)
        possible_labels = filter(lambda i: self.nodes[i]['distance'] == min_distance, keys)
        if len(possible_labels) > 0:
            lb = possible_labels[0]
            self.nodes[lb]['chosen'] = True
            self.nodes[lb]['allowed'] = False
        else:
            lb = None
 
        return lb
            
    def build_graph(self, adjmat, props):
        self.nodes = {}
        for i in range(1, adjmat.shape[0]):
            lab_vec = np.where(adjmat[i,:])[0]            
            self.nodes[i] = {'allowed': True,
                             'neighbors': lab_vec[lab_vec>0], 
                             'distance': props[i]['distance'],
                             'chosen': False}
        return
    
    # recursive expansion
    def rec_expansion(self, lb, k):
        if k==0:
            return
        neighbors = self.nodes[lb]['neighbors']
        for nb in neighbors:            
            self.nodes[nb]['allowed'] = False
            self.expansion(nb, k-1)
        return
        
    def expansion_one_step(self, lb):
        changed=[]
        neighbors = self.nodes[lb]['neighbors']
        for nb in neighbors:            
            if self.nodes[nb]['allowed']:
                changed.append(nb)
            self.nodes[nb]['allowed'] = False
                    
        return changed
    
    def expansion(self, lb, K):
        
        labels_todo = [lb]
        next_step = []
        for i in range(K):
            for lb in labels_todo:
                changed = self.expansion_one_step(lb)
                next_step.extend(changed)
            labels_todo = next_step
            next_step = []

        candidates = []                
        for lb in labels_todo:
            candidates.extend(self.nodes[lb]['neighbors'].tolist())
        candidates = list(set(candidates))
        candidates = filter(lambda x: self.nodes[x]['allowed'], candidates)
        return candidates
        
    
    def get_chosen_labels(self):
        chosen = filter(lambda x: self.nodes[x]['chosen'], self.nodes.keys())
        return chosen
    
    def __call__(self, imin, imbin, K):
        labels = self.find_neighbors(imbin, 100)
        cooc = skimage.feature.greycomatrix(labels, [1], 
                                            [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                            levels=labels.max() + 1)
        cooc = np.sum(cooc[:,:,0,:], axis=2)
        cooc = cooc - np.diag(np.diag(cooc))
        adjmat = cooc>0
        
        node_properties = self.get_properties(imbin, imin)
        self.distance_to_avg(node_properties)        
        self.build_graph(adjmat, node_properties)
        
        for node in sorted(self.nodes.keys()):
            print node, self.nodes[node]
        
        # initialization
        lb = self.select_label()
        self.nodes[lb]['chosen'] = True
        self.nodes[lb]['allowed'] = False
        
        # loop
        while not lb is None:
            labels_todo = self.expansion(lb, K)            
            
#             print
#             print labels_todo
#             print [self.nodes[x]['allowed'] for x in labels_todo]
            
            if len(labels_todo) > 0:                
                lb = self.select_label(labels_todo)
                self.nodes[lb]['chosen'] = True
                self.nodes[lb]['allowed'] = False
            else:
                lb = None
                
            if lb is None:
                lb = self.select_label()
#             print lb
#             if lb==96:
#                 pdb.set_trace()
                
        
        chosen_labels = self.get_chosen_labels()
        print chosen_labels
        all_labels = np.zeros(labels.max() + 1)
        all_labels += 100
        all_labels[0] = 0
        for l in chosen_labels:
            all_labels[l] = 255
        
        cell_labels = label(imbin, neighbors=4)
        res = all_labels[cell_labels]
        
        out_filename = os.path.join(self.settings.cell_selection_folder, 'test.png')
        skimage.io.imsave(out_filename, res.astype(np.dtype('uint8')))

        return
    

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
            channel_img[imvis>0] *= (1-alpha) 
            
            colim[:,:,i] = alpha*col*imvis + channel_img
        
        return colim
    
class SimpleWorkflow(object):
    
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

        self.ov = Overlay()
        
        # for VIGRA
        #self.ut = utilities.Utilities()
                              
    def __call__(self, img_16bit):
        
        print '8 bit --> 16 bit'
        img = self.reduce_range(img_16bit)
        
        print 'prefiltering'
        pref = self.prefilter(img, self.settings.segmentation_settings['prefiltering'])
        
        if self.settings.debug:
            filenames = os.listdir(self.settings.debug_folder)
            if len(filenames) == 0:
                index = 0
            else:
                already_done = [int(os.path.splitext(x)[0][:3]) for x in 
                                filter(lambda y: os.path.splitext(y)[-1].lower() in ['.tif', '.tiff', '.png'] and y[0] in ['0', '1', '2'], filenames)]
                if len(already_done) == 0:
                    index = 1
                else:
                    index = np.max(already_done) + 1
                
            out_filename = os.path.join(self.settings.debug_folder, '%03i__01_original.png' % index) 
            skimage.io.imsave(out_filename, img)

            out_filename = os.path.join(self.settings.debug_folder, '%03i__02_prefiltered.png' % index)                                                                                
            skimage.io.imsave(out_filename, pref)
            
        
        bgsub = self.background_subtraction(pref, self.settings.segmentation_settings['bg_sub'])
        hmax = self.homogenize(pref)
        
        if self.settings.debug:
            out_filename = os.path.join(self.settings.debug_folder, '%03i__03_bgsub.png' % index)                                                                                
            skimage.io.imsave(out_filename, bgsub)

            out_filename = os.path.join(self.settings.debug_folder, '%03i__04_hmax.png' % index)                                                                                
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

        #pdb.set_trace()
        if self.settings.debug:
            out_filename = os.path.join(self.settings.debug_folder, '%03i__05_local_adaptive_threshold.png' % index)                                                                                
            skimage.io.imsave(out_filename, segmentation_result)

            out_filename = os.path.join(self.settings.debug_folder, '%03i__06_seghmax.png' % index)                                                                                
            skimage.io.imsave(out_filename, seg_hmax)

        segmentation_result[seg_hmax>0] = 255
                
        if self.settings.debug:            
            out_filename = os.path.join(self.settings.debug_folder, '%03i__07_segres.png' % index)                                                                                
            skimage.io.imsave(out_filename, segmentation_result)

            overlay_img = self.ov.to_gray_scale(img, segmentation_result, (1.0, 0.0, 0.0), 0.8)
            out_filename = os.path.join(self.settings.debug_folder, '%03i__08_overlay_both.png' % index)                                                                                
            skimage.io.imsave(out_filename, overlay_img)
                    
        # split
        labres = self.split_cells(segmentation_result, img)
        wsl = self.filter_wsl(segmentation_result, labres, img)
        #wsl = self.get_internal_wsl(labres)
        res = segmentation_result.copy()
        res[wsl>0] = 0

        if self.settings.debug:
            out_filename = os.path.join(self.settings.debug_folder, '%03i__09_after_split.png' % index)
            skimage.io.imsave(out_filename, res)

            overlay_img = self.ov.to_gray_scale(img, res, (1.0, 0.0, 0.0), 0.8)
            out_filename = os.path.join(self.settings.debug_folder, '%03i__10_overlay_after_split.png' % index)                                                                                
            skimage.io.imsave(out_filename, overlay_img)
        
        # postfiltering
        res = self.postfilter(res, img)

        if self.settings.debug:
            out_filename = os.path.join(self.settings.debug_folder, '%03i__11_postfilter.png' % index)
            skimage.io.imsave(out_filename, res)

            overlay_img = self.ov.to_gray_scale(img, res, (1.0, 0.0, 0.0), 0.8)
            out_filename = os.path.join(self.settings.debug_folder, '%03i__12_overlay_postfilter.png' % index)                                                                                
            skimage.io.imsave(out_filename, overlay_img)

            final_label = label(res, neighbors=4)
            out_filename = os.path.join(self.settings.debug_folder, '%03i__12bis_labels_after_postfilter.png' % index)
            skimage.io.imsave(out_filename, final_label)

        res = self.remove_border_objects(res)

        if self.settings.debug:
            out_filename = os.path.join(self.settings.debug_folder, '%03i__13_border_obj_removed.png' % index)
            skimage.io.imsave(out_filename, res)

            overlay_img = self.ov.to_gray_scale(img, res, (1.0, 0.0, 0.0), 0.8)
            out_filename = os.path.join(self.settings.debug_folder, '%03i__14_overlay_border_obj_removed.png' % index)                                                                                
            skimage.io.imsave(out_filename, overlay_img)

            final_label = label(res, neighbors=4)
            out_filename = os.path.join(self.settings.debug_folder, '%03i__15_final_label.png' % index)
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
    
    def remove_border_objects(self, imbin):
        labelim = label(imbin, neighbors=4)
        #pdb.set_trace()
        
        A = np.hstack([labelim[0,:], labelim[-1,:], labelim[:,0], labelim[:,-1]])
        border_counts = np.bincount(A)
        if len(border_counts) < labelim.max() + 1:
            delta = labelim.max() + 1 - len(border_counts)
            border_counts = np.hstack([border_counts, np.zeros(delta)])            
        
        border_filter = np.where(border_counts > 0, 255, 0)
        
        to_remove = border_filter[labelim]        
        imbin[to_remove>0] = 0        
        
        return imbin
    
    def filter_wsl(self, imbin, ws_labels, imin):
        
        # internal gradient of the cells: 
        se = morphology.diamond(1)
        ero = morphology.erosion(imbin, se)        
        grad = imbin - ero
        
        # watershed line        
        wsl = self.get_internal_wsl(ws_labels)
        wsl_remove = wsl.copy()

        # watershed line outside the cells is 0
        wsl_remove[imbin==0] = 0
        # watershed line on the gradient (border of objects)
        # is also not considered
        wsl_remove[grad>0] = 0
                
        # gradient image
        pref = 255 * filters.gaussian_filter(imin, 3.0)
        pref[pref < 0] = 0
        pref = pref.astype(np.dtype('uint8'))
        ero = morphology.erosion(pref, se)
        dil = morphology.dilation(pref, se)
        grad = dil - ero
        grad_filtered = grad
        
        if self.settings.debug:
            out_filename = os.path.join(self.settings.debug_folder, 'wsl.png')
            skimage.io.imsave(out_filename, wsl.astype(np.dtype('uint8')))        

            out_filename = os.path.join(self.settings.debug_folder, 'wsl_remove.png')
            skimage.io.imsave(out_filename, wsl_remove.astype(np.dtype('uint8')))        

            out_filename = os.path.join(self.settings.debug_folder, 'wsl_gradient.png')
            
            skimage.io.imsave(out_filename, grad_filtered.astype(np.dtype('uint8')))        
        
        labimage = label(wsl_remove)
        properties = measure.regionprops(labimage, grad_filtered)   
        
        mean_intensities = np.array([0.0] + [pr.mean_intensity for pr in properties])
        filter_intensities = np.where(mean_intensities < self.settings.postfilter['wsl_mean_intensity'], 255, 0)
        filter_intensities[0] = 0
        
        wsl_remove = filter_intensities[labimage]
        print filter_intensities
        print mean_intensities
        wsl[wsl_remove>0] = 0

        if self.settings.debug:
            out_filename = os.path.join(self.settings.debug_folder, 'wsl_remove2.png')
            skimage.io.imsave(out_filename, wsl_remove.astype(np.dtype('uint8')))        

        return wsl
    
    def split_cells(self, segres, imin):
        ssc = self.settings.split_cells
        
        distance = ndi.distance_transform_edt(segres)
        
        distance_filtered = filters.gaussian_filter(distance, ssc['sigma'])
        local_maxima = self.local_maxima(distance_filtered, ssc['h'])
        #local_maxima = peak_local_max(distance_filtered, indices=False, 
        #                              footprint=np.ones((3, 3)), 
        #                              min_distance=20)
        #local_maxima[local_maxima>0] = 255
        
        if self.settings.debug:
            out_filename = os.path.join(self.settings.debug_folder, 'distance_function.png')
            skimage.io.imsave(out_filename, distance.astype(np.dtype('uint8')))

            out_filename = os.path.join(self.settings.debug_folder, 'local_maxima.png')
            lm = local_maxima.astype(np.dtype('uint8'))
            lm[lm>0] = 255
            skimage.io.imsave(out_filename, lm)
            

        #local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
        #                            labels=image)
        dist_img = distance.max() - distance
        dist_img = dist_img.astype(np.dtype('int64'))
        markers = label(local_maxima)
        labels = watershed(dist_img, markers, mask=segres)
        
        return labels
    
    def postfilter(self, segres, img):
        labimage = label(segres)
        properties = measure.regionprops(labimage, img)        
        
        areas = np.array([0] + [pr.area for pr in properties])
        filter_areas = np.where(areas > self.settings.postfilter['area'], 255, 0)
        #res = filter_areas[labimage]
        
        mean_intensities = np.array([0.0] + [pr.mean_intensity for pr in properties])
        filter_intensities = np.where(mean_intensities > self.settings.postfilter['mean_intensity'], 255, 0)

        total_filter = np.min(np.vstack([filter_areas, filter_intensities]), axis=0)
        res = total_filter[labimage]
        
        return res
        
    def reduce_range(self, img):
        minval = np.min(img)
        temp = 255.0 / 2**16 * (img - np.min(img))
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
                                                 mode='constant', cval=0)
                       
        elif method=='close_rec':
            se = disk(ps['close_size'])
            dil = morphology.dilation(img, se)
            rec = morphology.reconstruction(dil, img, method='erosion')
            
            # reconstruction gives back a float image (for whatever reason). 
            pref = rec.astype(dil.dtype)
            
        elif method=='denbi_clorec':
            temp = restoration.denoise_bilateral(img, ps['win_size'], ps['sigma_signal'], ps['sigma_space'], ps['bins'], 
                                                 mode='constant', cval=0)
            temp = 255 * temp
            temp = temp.astype(img.dtype)
            
            se = disk(ps['close_size'])
            dil = morphology.dilation(temp, se)
            rec = morphology.reconstruction(dil, temp, method='erosion')
            
            # reconstruction gives back a float image (for whatever reason). 
            pref = rec.astype(img.dtype)            

        elif method=='denbi_asfrec':
            temp = restoration.denoise_bilateral(img, ps['win_size'], ps['sigma_signal'], ps['sigma_space'], ps['bins'], 
                                                 mode='constant', cval=0)
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
            radius = ps['median_size']
            pref = rank.median(img, disk(radius))

            temp = restoration.denoise_bilateral(pref, ps['win_size'], ps['sigma_signal'], ps['sigma_space'], ps['bins'], 
                                                 mode='constant', cval=0)
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

        
    def __call__(self):
        filenames = filter(lambda x: os.path.splitext(x)[-1] in ['.tif', '.tiff', '.TIFF', '.TIF'], 
                           os.listdir(self.settings.data_folder))
        for filename in filenames[:1]:
            
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
            skimage.io.imsave(out_filename, res)
            
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
    
    
        