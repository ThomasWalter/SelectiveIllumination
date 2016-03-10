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

from Queue import Queue

import pdb
#from test.test_support import temp_cwd
#from rdflib.plugins.parsers.pyRdfa.transform.prototype import pref

import skimage.draw
#from jinja2.nodes import Pos

class Select(object):
    def __init__(self, settings_filename=None, settings=None, prefix=''):
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
        self.prefix=prefix
        
    def overlay_graph(self, img, labels=None):
        thickness = 3
        radius_offset=2

        if labels is None:
            labels = []
    
        #img = self.sw.reduce_range(img_16bit)
             
        colim = color.gray2rgb(img)
        if self.nodes is None:
            print 'graph is not yet built.'
            return
        
                   
#         max_color = max(colorvalue)
#         if max_color > 1.0:
#             colorvalue = [a / np.float(max_color) for a in colorvalue]
            
        # first we draw the edges
        for node_id, node in self.nodes.iteritems():                           
            for nb_id in node['neighbors']:
                nb_center = self.nodes[nb_id]['center']
                
                rr, cc = skimage.draw.line(np.int(np.round(node['center'][0])), 
                                           np.int(np.round(node['center'][1])),
                                           np.int(np.round(self.nodes[nb_id]['center'][0])), 
                                           np.int(np.round(self.nodes[nb_id]['center'][1])) )
                    
                colorvalue = (255,255,255)
                for i, col in enumerate(colorvalue):
                    colim[rr,cc,i] = col
            
                    
        #pdb.set_trace()
        
        # second, we draw the graph labels
        for nodelabel, node in self.nodes.iteritems():                                           
            rr, cc = skimage.draw.circle(np.round(node['center'][0]), 
                                        np.round(node['center'][1]), 
                                        self.settings.graph_radius)
            
            if node['chosen']:
                colorvalue = self.settings.graph_color_code[0]
            elif node['allowed']: 
                colorvalue = self.settings.graph_color_code[1]
            else:  
                colorvalue = self.settings.graph_color_code[2]
                             
            for i, col in enumerate(colorvalue):
                colim[rr,cc,i] = col
 
            if nodelabel in labels:
                colorvalue = labels[nodelabel]
                for cocentric in range(thickness):
                    rr, cc = skimage.draw.circle_perimeter(np.int(np.round(node['center'][0])), 
                                                           np.int(np.round(node['center'][1])), 
                                                           np.int(self.settings.graph_radius+radius_offset+cocentric),
                                                           method='andres')
                
                    for i, col in enumerate(colorvalue):
                        colim[rr,cc,i] = col
                
        return colim
        
    def find_neighbors(self, imbin, max_extension):
        background = np.zeros(imbin.shape)
        background[imbin==0] = 255
        distance = ndi.distance_transform_edt(background)
        cell_labels = label(imbin, neighbors=4, background=0)

        # this is a hack, as background pixels obtain label -1
        # and we do not want to have negative values. 
        # from version 0.12 this can be removed, probably.
        cell_labels = cell_labels - cell_labels.min()
        
        # the mask is an extension of the initial shape by max_extension. 
        # it can be derived from the distance map (straight forward)
        mask = np.zeros(imbin.shape)
        mask[distance < max_extension] = 255

        # The watershed of the distance transform of the background. 
        # this corresponds an approximation of the "cells"
        labels = watershed(distance, cell_labels, mask=mask)

        if self.settings.debug:
            out_filename = os.path.join(self.settings.cell_selection_folder, '%sdistance.png' % self.prefix)
            temp = distance / distance.max()
            skimage.io.imsave(out_filename, temp)

            out_filename = os.path.join(self.settings.cell_selection_folder, '%slabels_from_ws.png' % self.prefix)
            skimage.io.imsave(out_filename, labels)        
        
        return labels
    
    def get_properties(self, imbin, img):
        props = {}
        cell_labels = label(imbin, neighbors=4, background=0)
        cell_labels = cell_labels - cell_labels.min()
        properties = measure.regionprops(cell_labels, img)
        areas = [0] + [pr.area for pr in properties]
        convex_areas = [1] + [pr.convex_area for pr in properties]
        mean_intensities = [0.0] + [pr.mean_intensity for pr in properties]        
        eccentricities = [0.0] + [pr.eccentricity for pr in properties]
        centers = [(0.0, 0.0)] + [pr.centroid for pr in properties]
        perimeters = [1.0] + [pr.perimeter for pr in properties]
        a = np.array(areas)
        b = np.array(perimeters)
        b[b==0.0] = 1.0
        circ = 2 * np.sqrt(np.pi) * a / b
        c = np.array(convex_areas)
        cc_ar = a.astype(np.dtype('float')) / c.astype(np.dtype('float'))
        for i in range(1, cell_labels.max()+1):
            props[i] = {
                        'area': areas[i],
                        'mean_intensity': mean_intensities[i],
                        'eccentricity': eccentricities[i],
                        'center': centers[i],
                        'circularity': circ[i], 
                        'cc_ar': cc_ar[i],
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

        mean_circ = np.mean([props[i]['circularity'] for i in props.keys()])
        std_circ = np.std([props[i]['circularity'] for i in props.keys()])

        mean_cc_ar = np.mean([props[i]['cc_ar'] for i in props.keys()])
        std_cc_ar = np.std([props[i]['cc_ar'] for i in props.keys()])
        
        if std_area==0.0: std_area = 1.0
        if std_intensity==0.0: std_intensity = 1.0
        if std_ecc==0.0: std_ecc = 1.0
        if std_circ==0.0: std_circ = 1.0
        if std_cc_ar==0.0: std_cc_ar = 1.0
            
        for i in props.keys(): 
#             temp = (props[i]['area'] - mean_area)**2 + \
#                    (props[i]['mean_intensity'] - mean_intensity)**2 + \
#                    (props[i]['eccentricity'] - mean_ecc)**2
            temp = ((props[i]['area'] - mean_area) / std_area)**2 + \
                   ((props[i]['mean_intensity'] - mean_intensity) / std_intensity)**2 + \
                   ((props[i]['eccentricity'] - mean_ecc) / std_ecc)**2 + \
                   ((props[i]['cc_ar'] - mean_cc_ar) / std_cc_ar)**2 + \
                   ((props[i]['circularity'] - mean_circ) / std_circ)**2 

            props[i]['distance'] = np.sqrt(temp)
                
        return props
    
    def old_select_label(self, labels=None):
        
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

    def select_label(self, labels=None):
        
        if labels is None:
            keys = sorted(filter(lambda x: self.nodes[x]['allowed'], self.nodes.keys()))
        elif len(labels) > 0:
            keys = sorted(filter(lambda x: self.nodes[x]['allowed'], labels))
        else:
            keys = []
        if len(keys) == 0:
            return None
        
        distances = np.array([self.nodes[i]['distance'] for i in keys])
        
        min_distance = np.min(distances)
        possible_labels = filter(lambda i: self.nodes[i]['distance'] == min_distance, keys)
        if len(possible_labels) > 0:
            lb = possible_labels[0]
        else:
            lb = None
 
        #pdb.set_trace()
        return lb
            
    def build_graph(self, adjmat, props):
        self.nodes = {}
        for i in range(1, adjmat.shape[0]):
            lab_vec = np.where(adjmat[i,:])[0]          
            self.nodes[i] = {'allowed': True,
                             'neighbors': lab_vec[lab_vec>0], 
                             'distance': props[i]['distance'],
                             'center': props[i]['center'],
                             'chosen': False,
                             'min_dist': adjmat.shape[0] + 1}
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
    
    def choose_labels(self, K, imin_param=None):        
        if self.settings.debug:
            for node_id, node in self.nodes.iteritems():
                print '%i: %s' % (node_id, self.nodes[node_id]['neighbors'])

        imin = imin_param            
        if not imin_param is None:
            if not imin_param.dtype == 'uint8': 
                imin = self.sw.reduce_range(imin_param)
            
        # initialization
        Lres = []
        lb = self.select_label()
        #self.nodes[lb]['chosen'] = True
        #self.nodes[lb]['min_dist'] = 0
        #self.nodes[lb]['allowed'] = False
        #Lcand = self.nodes[lb]['neighbors'].tolist()
        Lcand = [lb]
        
        #for nb in self.nodes[lb]['neighbors']:
        #    Lcand.append(nb)
            
        r=0
        
        while len(Lcand) > 0:
            
            # gets the best label among the candidates
            # the best means the closest to the average.
            v = self.select_label(Lcand)
            
            if self.settings.debug:                
                if not v is None:
                    print 'Candidate: %i\tlabel: %i from %s' % (r, v, str(Lcand))
                else: 
                    print 'no suitable candidate from candidate list'
                    
            if v is None:
                # if none of the candidates is suitable, 
                # we enlarge the search to all nodes.
                v = self.select_label()
                
            if v is None:
                # this means that there is no suitable candidate. 
                Lcand = []
                break
            
            # v has been selected.
            if self.settings.debug and not imin is None:
                colim = self.overlay_graph(imin, labels=dict(zip(Lcand, [(230, 200, 0) for ttt in Lcand])))
                skimage.io.imsave(os.path.join(self.settings.debug_graph_overlay, '%sgraph_overlay_candidate%03i_a.png' % (self.prefix, r)), colim)
            self.nodes[v]['chosen'] = True
            if self.settings.debug and not imin is None:
                colim = self.overlay_graph(imin, labels=dict(zip(Lcand, [(230, 200, 0) for ttt in Lcand])))
                skimage.io.imsave(os.path.join(self.settings.debug_graph_overlay, '%sgraph_overlay_candidate%03i_b.png' % (self.prefix, r)), colim)

            Lres.append(v)

            Q1 = Queue()
            Q2 = Queue()
                        
            Q1.put(v)

            # expansion step
            for k in range(K+1): 
                if self.settings.debug and k>0:
                    print '\texpansion: node %i\texpansion step %i\tcandidate: %i' % (v, k, r)

                while not Q1.empty():

                    n = Q1.get()
                    self.nodes[n]['min_dist'] = min(self.nodes[n]['min_dist'], k)
                    self.nodes[n]['allowed'] = False

                    if self.settings.debug:
                        print '\t\t--> from Q1: %i' % n
                    #pdb.set_trace()    
                                            
                    for nb in self.nodes[n]['neighbors']:
                        if self.nodes[nb]['min_dist'] > k: 
                            Q2.put(nb)

                if self.settings.debug and not imin is None and k>0:
                    lc_filtered = filter(lambda x: self.nodes[x]['allowed'], Lcand)
                    colim = self.overlay_graph(imin, labels=dict(zip(lc_filtered, [(230, 200, 0) for ttt in lc_filtered])))
                    skimage.io.imsave(os.path.join(self.settings.debug_graph_overlay, '%sgraph_overlay_candidate%03i_step%03i.png' % (self.prefix, r, k)), colim)
                
                Q1 = Q2
                Q2 = Queue()
                                        
            # add new candidates
            while not Q1.empty():
                n = Q1.get()
                if self.nodes[n]['allowed']:
                    Lcand.append(n)
            
            if len(Lcand) == 0:
                v = self.select_label()
                if not v is None:
                    Lcand = [v]

            Lcand = list(set(filter(lambda x: x != v and self.nodes[x]['allowed'], Lcand)))
            
            r += 1
            
        return Lres
        
    def centers_to_text_file(self, imout, filename):
        # get the centers
        labels = label(imout, neighbors=4, background=0)
        properties = measure.regionprops(labels, imout)
        centers = [(0.0, 0.0)] + [pr.centroid for pr in properties]
        
        fp = open(filename, 'w')
        for x,y in centers[1:]:
            fp.write('%i\t%i\n' % (x,y))
        fp.close()
        
        return

    def circles_to_xml_file(self, imout, filename):
        # get the centers
        labels = label(imout, neighbors=4, background=0)
        properties = measure.regionprops(labels, imout)
        centers = [(0.0, 0.0)] + [pr.centroid for pr in properties]
        
        
        fp = open(filename, 'w')
        for x,y in centers[1:]:
            fp.write('%i\t%i\n' % (x,y))
        fp.close()
        
        return
        
    def __call__(self, imin, imbin, K):
        
        # First we label cellular regions. 
        # this can be done by a simple voronoi approach or by some other method.
        # here it is done with voronoi (with max extension of 100). 
        labels = self.find_neighbors(imbin, 100)

        if self.settings.debug:
            skimage.io.imsave(os.path.join(self.settings.debug_folder, 
                                           '%sbasis_for_graph.png' % self.prefix), labels)
        
        # find the co-occurence matrix. 
        # The co-occurence matrix informs us about the neighboring relationships.
        # From there, we can build the graph.
        cooc = skimage.feature.greycomatrix(labels, [1], 
                                            [0, np.pi/4, np.pi/2, 3*np.pi/4,
                                             np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4], 
                                            levels=labels.max() + 1)

        # we sum over different directions
        cooc = np.sum(cooc[:,:,0,:], axis=2)
        # and we remove the diagonal
        cooc = cooc - np.diag(np.diag(cooc))
        
        # adjacency matrix corresponds to the entries > 0.
        adjmat = cooc>0
                
        # calculate properties to find the most "representative cell" 
        node_properties = self.get_properties(imbin, imin)
        self.distance_to_avg(node_properties)

        # build graph
        self.build_graph(adjmat, node_properties)
         
        # select the cells (main algorithm)
        labres = self.choose_labels(K, imin)

        # get an output image
        label_values = np.zeros((labels.max() + 1)).astype(np.uint8)
        label_values[np.array(labres)] = 255
        imout = label_values[labels]
        
        return imout
    

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
    
    def __init__(self, settings_filename=None, settings=None, prefix=''):
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
        self.prefix=prefix
                              
    def __call__(self, img_16bit):
        
        print '16 bit --> 8 bit'
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
                
            out_filename = os.path.join(self.settings.debug_folder, '%s%03i__01_original.png' % (self.prefix, index) )
            skimage.io.imsave(out_filename, img)

            out_filename = os.path.join(self.settings.debug_folder, '%s%03i__02_prefiltered.png' % (self.prefix, index))                                                                                
            skimage.io.imsave(out_filename, pref)
            
        
        bgsub = self.background_subtraction(pref, self.settings.segmentation_settings['bg_sub'])
        hmax = self.homogenize(pref)
        
        if self.settings.debug:
            out_filename = os.path.join(self.settings.debug_folder, '%s%03i__03_bgsub.png' % (self.prefix, index))                                                                                
            skimage.io.imsave(out_filename, bgsub)

            out_filename = os.path.join(self.settings.debug_folder, '%s%03i__04_hmax.png' % (self.prefix, index))                                                                                
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
            out_filename = os.path.join(self.settings.debug_folder, '%s%03i__05_local_adaptive_threshold.png' % (self.prefix, index))                                                                                
            skimage.io.imsave(out_filename, segmentation_result)

            out_filename = os.path.join(self.settings.debug_folder, '%s%03i__06_seghmax.png' % (self.prefix, index))                                                                                
            skimage.io.imsave(out_filename, seg_hmax)

        segmentation_result[seg_hmax>0] = 255
                
        if self.settings.debug:            
            out_filename = os.path.join(self.settings.debug_folder, '%s%03i__07_segres.png' % (self.prefix, index))                                                                                
            skimage.io.imsave(out_filename, segmentation_result)

            overlay_img = self.ov.to_gray_scale(img, segmentation_result, (1.0, 0.0, 0.0), 0.8)
            out_filename = os.path.join(self.settings.debug_folder, '%s%03i__08_overlay_both.png' % (self.prefix, index))                                                                                
            skimage.io.imsave(out_filename, overlay_img)
                    
        # split
        labres = self.split_cells(segmentation_result, img)
        wsl = self.filter_wsl(segmentation_result, labres, img)
        #wsl = self.get_internal_wsl(labres)
        res = segmentation_result.copy()
        res[wsl>0] = 0

        if self.settings.debug:
            out_filename = os.path.join(self.settings.debug_folder, '%s%03i__09_after_split.png' % (self.prefix, index))
            skimage.io.imsave(out_filename, res)

            overlay_img = self.ov.to_gray_scale(img, res, (1.0, 0.0, 0.0), 0.8)
            out_filename = os.path.join(self.settings.debug_folder, '%s%03i__10_overlay_after_split.png' % (self.prefix, index))                                                                                
            skimage.io.imsave(out_filename, overlay_img)
        
        # postfiltering
        res = self.postfilter(res, img)

        if self.settings.debug:
            out_filename = os.path.join(self.settings.debug_folder, '%s%03i__11_postfilter.png' % (self.prefix, index))
            skimage.io.imsave(out_filename, res)

            overlay_img = self.ov.to_gray_scale(img, res, (1.0, 0.0, 0.0), 0.8)
            out_filename = os.path.join(self.settings.debug_folder, '%s%03i__12_overlay_postfilter.png' % (self.prefix, index))                                                                                
            skimage.io.imsave(out_filename, overlay_img)

            final_label = label(res, neighbors=4, background=0)
            final_label = final_label - final_label.min()
            out_filename = os.path.join(self.settings.debug_folder, '%s%03i__12bis_labels_after_postfilter.png' % (self.prefix, index))
            skimage.io.imsave(out_filename, final_label)

        res = self.remove_border_objects(res)

        if self.settings.debug:
            out_filename = os.path.join(self.settings.debug_folder, '%s%03i__13_border_obj_removed.png' % (self.prefix, index))
            skimage.io.imsave(out_filename, res)

            overlay_img = self.ov.to_gray_scale(img, res, (1.0, 0.0, 0.0), 0.8)
            out_filename = os.path.join(self.settings.debug_folder, '%s%03i__14_overlay_border_obj_removed.png' % (self.prefix, index))                                                                                
            skimage.io.imsave(out_filename, overlay_img)

            final_label = label(res, neighbors=4, background=0)
            final_label = final_label - final_label.min()
            out_filename = os.path.join(self.settings.debug_folder, '%s%03i__15_final_label.png' % (self.prefix, index))
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
        labelim = label(imbin, neighbors=4, background=0)
        labelim = labelim - labelim.min()
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
            out_filename = os.path.join(self.settings.debug_folder, '%swsl_remove2.png' % self.prefix)
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
            out_filename = os.path.join(self.settings.debug_folder, '%sdistance_function.png' % self.prefix)
            skimage.io.imsave(out_filename, distance.astype(np.dtype('uint8')))

            out_filename = os.path.join(self.settings.debug_folder, '%slocal_maxima.png' % self.prefix)
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
    
    
        