import os

from segmentation.basic import SimpleWorkflow
from segmentation.basic import Overlay

from settings import *

import numpy as np

# skimage imports
import skimage
import skimage.io
import skimage.draw
from skimage import color
from skimage.measure import label
from skimage import measure
from skimage.feature import peak_local_max
from skimage.morphology import watershed

from scipy import ndimage as ndi

from Queue import Queue

import pdb

from PIL import Image
import xml.etree.ElementTree as ET

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
    
        width = img.shape[1]
        height = img.shape[0]
        
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

                indices = filter(lambda i: rr[i] > 0 and rr[i] < height and cc[i]>0 and cc[i]<width, 
                                 range(len(rr)) )
                rr = rr[indices]
                cc = cc[indices]
            
                for i, col in enumerate(colorvalue):
                    colim[rr,cc,i] = col
            
        # second, we draw the graph labels
        for nodelabel, node in self.nodes.iteritems():                                           
            rr, cc = skimage.draw.circle(np.round(node['center'][0]), 
                                        np.round(node['center'][1]), 
                                        self.settings.graph_radius)
            indices = filter(lambda i: rr[i] > 0 and rr[i] < height and cc[i]>0 and cc[i]<width, 
                             range(len(rr)) )
            rr = rr[indices]
            cc = cc[indices]

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
                
                    indices = filter(lambda i: rr[i] > 0 and rr[i] < height and cc[i]>0 and cc[i]<width, 
                                     range(len(rr)) )
                    rr = rr[indices]
                    cc = cc[indices]

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

        if self.settings.debug_screen_output:
            out_filename = os.path.join(self.settings.img_debug_folder, '%s16_distance.png' % self.prefix)
            temp = distance / distance.max()
            skimage.io.imsave(out_filename, temp)

            out_filename = os.path.join(self.settings.img_debug_folder, '%s16_labels_from_ws.png' % self.prefix)
            skimage.io.imsave(out_filename, labels)        
        
        return cell_labels, labels
    
    def read_header(self, filename):
        
        impil = Image.open(filename)  
          
        tagged_string = impil.tag.tagdata[270]
        root = ET.fromstring(tagged_string.strip('\x00'))
        subtree = root.find('PlaneInfo')
        properties = subtree.findall('prop')
        #<prop id="stage-position-x" type="float" value="18988"/>
        #<prop id="stage-position-y" type="float" value="15355"/>
        #<prop id="stage-label" type="string" value=""/>
        #<prop id="z-position" type="float" value="-1392.99"/>

        x = None
        y = None
        z = None
        for pr in properties: 
            if 'id' in pr.attrib and pr.attrib['id'] == 'z-position':
                z = float(pr.attrib['value'])
            if 'id' in pr.attrib and pr.attrib['id'] == 'stage-position-x':
                x = float(pr.attrib['value'])
            if 'id' in pr.attrib and pr.attrib['id'] == 'stage-position-y':
                y = float(pr.attrib['value'])
        
        print 'x %f, y %f, z %f ' % (x, y, z)
        return x, y, z

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
 
        return lb

    def eudist(self, a, b):
        res = np.sqrt(np.sum((np.array(a) - np.array(b))**2))
        return res
    
    
    def normalize_vec_eu(self, a):
        mean_a = np.mean(a)
        std_a = np.std(a)

        if std_a > 0:
            norm_vec = (a - mean_a) / std_a
        else:
            norm_vec = a - mean_a

        return norm_vec

    def normalize_vec(self, a):
        min_a = np.min(a)
        max_a = np.max(a)
        
        if max_a > min_a:
            norm_vec = (a.astype('float') - min_a) / (max_a - min_a)
        else:
            norm_vec = a.astype('float') - min_a

        return norm_vec
    
    def select_cluster_nodes(self, labels=None):
        
        if labels is None:
            # in this case we take all the allowed nodes
            keys = sorted(filter(lambda x: self.nodes[x]['allowed'], self.nodes.keys()))
        elif len(labels) > 0:
            # in this case we take only the given set of labels (but only the subset which is allowed). 
            keys = sorted(filter(lambda x: self.nodes[x]['allowed'], labels))
        else:
            return None        
        
        if len(keys) == 0:
            return None
        
        # among these labels, we need to find a seed for a cluster of cluster_size. 
        # this seed is found by minimizing the feature distance. As no seed is guaranteed
        # to result in a valid cluster, we take all valid nodes and rank them according to their distance to the mean cell. 
        if self.settings.ordered_cell_selection:            
            dist_to_avg = self.normalize_vec(np.array([self.nodes[i]['distance'] for i in keys]))
            np_connections = self.normalize_vec(np.array([len(self.nodes[i]['neighbors']) for i in keys]))
            x = self.normalize_vec(np.array([self.nodes[i]['center'][1] for i in keys]))
            y = self.normalize_vec(np.array([self.nodes[i]['center'][0] for i in keys]))
            scores = np.array(1000 * np_connections + 100 * x + 10 * y + dist_to_avg)
            indices = scores.argsort()
            ranked_keys = np.array(keys)[indices]
        else: 
            dist_to_avg = self.normalize_vec(np.array([self.nodes[i]['distance'] for i in keys]))
            #x = self.normalize_vec(np.array([self.nodes[i]['center'][1] for i in keys]))
            #y = self.normalize_vec(np.array([self.nodes[i]['center'][0] for i in keys]))
            np_connections = self.normalize_vec(np.array([len(self.nodes[i]['neighbors']) for i in keys]))
            
            scores = np.array(10 * (np.max(np_connections) - np_connections) + dist_to_avg)
            indices = scores.argsort()
            ranked_keys = np.array(keys)[indices]            
            
            
            #print 'node selection : ', labels, ' --> ', ranked_keys
            #print '\t', [len(self.nodes[i]['neighbors']) for i in keys]
            #print '\t', [len(self.nodes[i]['neighbors']) for i in ranked_keys]
            
            
        cluster_size = self.settings.cluster_size
        cluster_defined = False
        
        # loop over all potential cluster seeds
        while not cluster_defined and len(ranked_keys) > 0:

            # get the cluster seed (the best one remaining) 
            v = ranked_keys[0]
            ranked_keys = ranked_keys[1:]
            
            #cluster_nodes = [ranked_keys[0]]        
            cluster_nodes = []
            waiting_nodes = np.array([v])
            nb_chosen = len(cluster_nodes)

            # loop to expand the cluster seed to reach cluster_size if possible. 
            while nb_chosen < cluster_size and len(waiting_nodes) > 0:
                
                # for all waiting_nodes we calculate the maximal distance to the already selected nodes (cluster_nodes)
                # minimizing the maximal distance allows to have compact classes. 
                if len(waiting_nodes) > 1 and len(cluster_nodes) > 0:
                    eu_dist = [np.max(np.array([self.eudist(self.nodes[wo]['center'],
                                                            self.nodes[cn]['center']) 
                                                for cn in cluster_nodes])) 
                               for wo in waiting_nodes]
                    indices = np.array(eu_dist).argsort()
                    waiting_nodes = waiting_nodes[indices]
                    
                node_added = False
                while not node_added and len(waiting_nodes) > 0:
                    o = waiting_nodes[0]
                    waiting_nodes = waiting_nodes[1:]
                    
                    #if self.nodes[o]['allowed'] and not o in cluster_nodes: 
                    if not o in cluster_nodes:
                        cluster_nodes.append(o)
                        node_added = True
                        
                        # add the neighbors to the waiting_nodes
                        for nb in self.nodes[o]['neighbors']:

                            # neighbors are added 
                            if self.nodes[nb]['allowed'] and not nb in waiting_nodes and not nb in cluster_nodes:
                                waiting_nodes = np.append(waiting_nodes, nb)
                            
                nb_chosen = len(cluster_nodes)

            if len(cluster_nodes) < cluster_size:
                # in this case, it was not possible to expand the cluster sufficiently. We can therefore set the corresponding 
                # nodes to "not allowed"
                for node_id in cluster_nodes:
                    self.nodes[node_id]['allowed'] = False
            else:
                cluster_defined = True

        if not cluster_defined: 
            cluster_nodes = None
            
        return cluster_nodes
            
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
    
    def choose_labels_backup(self, K, imin_param=None):        
        if self.settings.debug_screen_output:
            for node_id, node in self.nodes.iteritems():
                print '%i: %s' % (node_id, self.nodes[node_id]['neighbors'])

        imin = imin_param            
        if not imin_param is None:
            if not imin_param.dtype == 'uint8': 
                imin = self.sw.reduce_range(imin_param, minmax=True)
            
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
            
            if self.settings.debug_screen_output:                
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
            if self.settings.graph_overlay and not imin is None:
                colim = self.overlay_graph(imin, labels=dict(zip(Lcand, [(230, 200, 0) for ttt in Lcand])))
                skimage.io.imsave(os.path.join(self.settings.img_graph_overlay_folder, '%sgraph_overlay_candidate%03i_a.png' % (self.prefix, r)), colim)
            self.nodes[v]['chosen'] = True
            if self.settings.graph_overlay and not imin is None:
                colim = self.overlay_graph(imin, labels=dict(zip(Lcand, [(230, 200, 0) for ttt in Lcand])))
                skimage.io.imsave(os.path.join(self.settings.img_graph_overlay_folder, '%sgraph_overlay_candidate%03i_b.png' % (self.prefix, r)), colim)

            Lres.append(v)

            Q1 = Queue()
            Q2 = Queue()
                        
            Q1.put(v)

            # expansion step
            for k in range(K+1): 
                if self.settings.debug_screen_output and k>0:
                    print '\texpansion: node %i\texpansion step %i\tcandidate: %i' % (v, k, r)

                while not Q1.empty():

                    n = Q1.get()
                    self.nodes[n]['min_dist'] = min(self.nodes[n]['min_dist'], k)
                    self.nodes[n]['allowed'] = False

                    if self.settings.debug_screen_output:
                        print '\t\t--> from Q1: %i' % n
                                            
                    for nb in self.nodes[n]['neighbors']:
                        if self.nodes[nb]['min_dist'] > k: 
                            Q2.put(nb)

                if self.settings.graph_overlay and not imin is None and k>0:
                    lc_filtered = filter(lambda x: self.nodes[x]['allowed'], Lcand)
                    colim = self.overlay_graph(imin, labels=dict(zip(lc_filtered, [(230, 200, 0) for ttt in lc_filtered])))
                    skimage.io.imsave(os.path.join(self.settings.img_graph_overlay_folder, '%sgraph_overlay_candidate%03i_step%03i.png' % (self.prefix, r, k)), colim)
                
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
    
    def choose_labels(self, K, imin_param=None):        
        if self.settings.debug_screen_output:
            for node_id, node in self.nodes.iteritems():
                print '%i: %s' % (node_id, self.nodes[node_id]['neighbors'])

        imin = imin_param            
        if not imin_param is None:
            if not imin_param.dtype == 'uint8': 
                imin = self.sw.reduce_range(imin_param, minmax=True)
            
        cluster_size = self.settings.cluster_size
        cluster_dist = self.settings.cluster_dist
        
        # initialization
        Lres = []
        if self.settings.ordered_cell_selection:                        
            #np_connections = self.normalize_vec(np.array([len(self.nodes[i]['neighbors']) for i in keys]))
            keys = sorted(self.nodes.keys())
            x = self.normalize_vec(np.array([self.nodes[i]['center'][1] for i in keys]))
            y = self.normalize_vec(np.array([self.nodes[i]['center'][0] for i in keys]))
            scores = np.array(x**2 + y**2)
            indices = scores.argsort()
            lb = keys[indices[0]]
        else:            
            lb = self.select_label()        

        Lcand = [lb]
        #Lcand = self.select_cluster_nodes()
        
        r=0
        
        while len(Lcand) > 0:
            
            # gets the best label among the candidates
            # the best means the closest to the average.
            #v = self.select_label(Lcand)
            v = self.select_cluster_nodes(Lcand)
            
#             if self.settings.debug_screen_output:                
#                 if not v is None:
#                     print 'Candidate: %i\tlabel: %i from %s' % (r, v, str(Lcand))
#                 else: 
#                     print 'no suitable candidate from candidate list'
                    
            if v is None:
                # if none of the candidates is suitable, 
                # we enlarge the search to all nodes.
                v = self.select_cluster_nodes()
                
            if v is None:
                # this means that there is no suitable candidate. 
                # the algorithm has to stop
                Lcand = []
                break
            
            # v has been selected.
            if self.settings.graph_overlay and not imin is None:
                colim = self.overlay_graph(imin, labels=dict(zip(Lcand, [(230, 200, 0) for ttt in Lcand])))
                skimage.io.imsave(os.path.join(self.settings.img_graph_overlay_folder, '%sgraph_overlay_candidate%03i_a.png' % (self.prefix, r)), colim)
            # all nodes in the cluster are set to "chosen"
            for s in v:
                self.nodes[s]['chosen'] = True
            if self.settings.graph_overlay and not imin is None:
                colim = self.overlay_graph(imin, labels=dict(zip(Lcand, [(230, 200, 0) for ttt in Lcand])))
                skimage.io.imsave(os.path.join(self.settings.img_graph_overlay_folder, '%sgraph_overlay_candidate%03i_b.png' % (self.prefix, r)), colim)

            Lres.extend(v)

            Q1 = Queue()
            Q2 = Queue()

            for node_id in v:                       
                Q1.put(node_id)

            # expansion step
            for k in range(K+1): 
                #if self.settings.debug_screen_output and k>0:
                #    print '\texpansion: node %i\texpansion step %i\tcandidate: %i' % (v, k, r)

                while not Q1.empty():

                    n = Q1.get()
                    self.nodes[n]['min_dist'] = min(self.nodes[n]['min_dist'], k)
                    self.nodes[n]['allowed'] = False

#                     if self.settings.debug_screen_output:
#                         print '\t\t--> from Q1: %i' % n
                                            
                    for nb in self.nodes[n]['neighbors']:
                        if self.nodes[nb]['min_dist'] > k: 
                            Q2.put(nb)

                if self.settings.graph_overlay and not imin is None and k>0:
                    lc_filtered = filter(lambda x: self.nodes[x]['allowed'], Lcand)
                    colim = self.overlay_graph(imin, labels=dict(zip(lc_filtered, [(230, 200, 0) for ttt in lc_filtered])))
                    skimage.io.imsave(os.path.join(self.settings.img_graph_overlay_folder, '%sgraph_overlay_candidate%03i_step%03i.png' % (self.prefix, r, k)), colim)
                
                
                Q1 = Q2
                Q2 = Queue()
                                        
            # add new candidates
            while not Q1.empty():
                n = Q1.get()
                if self.nodes[n]['allowed']:
                    Lcand.append(n)
            
            if len(Lcand) == 0:
                v = self.select_cluster_nodes()
                if not v is None:
                    Lcand = v

            Lcand = list(set(filter(lambda x: not self.nodes[x]['chosen'] and self.nodes[x]['allowed'], Lcand)))
            
            r += 1
            
        return Lres
        
    def centers_to_text_file(self, imout, filename):
        # get the centers
        labels = label(imout, neighbors=4, background=0)
        properties = measure.regionprops(labels, imout)
        #pdb.set_trace()
        centers = [(0.0, 0.0)] + [ (pr.centroid[0] * self.settings.param_pixel_size, 
                                    pr.centroid[1] * self.settings.param_pixel_size)
                                  for pr in properties]
        
        fp = open(filename, 'w')
        for x,y in centers[1:]:
            fp.write('%f\t%f\n' % (x,y))
        fp.close()
        
        return

    def centers_to_px_text_file(self, imout, filename):
        # get the centers
        labels = label(imout, neighbors=4, background=0)
        properties = measure.regionprops(labels, imout)
        centers = [(0.0, 0.0)] + [pr.centroid for pr in properties]
        
        fp = open(filename, 'w')
        for x,y in centers[1:]:
            fp.write('%i\t%i\n' % (x,y))
        fp.close()
        
        return

    def export_metamorph(self, imout, filename, stage_coord=None):
        
        if stage_coord is None:
            stage_coord = (0, 0, 0)
            
        # get the centers
        labels = label(imout, neighbors=4, background=0)
        properties = measure.regionprops(labels, imout)

        # check centers
        #for pr in properties:
        #    print 'image coordinates (pixel) : %i, %i' % (pr.centroid[0], pr.centroid[1])
            
        centers = [(0.0, 0.0)] + [ (pr.centroid[0] * self.settings.param_pixel_size, 
                                    pr.centroid[1] * self.settings.param_pixel_size)
                                  for pr in properties]
        nb_cells = len(centers) -1 

        # write the information in a metamorph .stg format.        
        fp = open(filename, 'w')
        fp.write('"Stage Memory List", Version 6.0\n')
        fp.write('0, 0, 0, 0, 0, 0, 0, "um", "um"\n')
        fp.write('0\n')
        fp.write('%i\n' % nb_cells)
        
        # image dimension in micrometer
        w = imout.shape[1] * self.settings.param_pixel_size
        h = imout.shape[0] * self.settings.param_pixel_size        
    
        print 'width and height : ', w, h    
        i = 1
        for y, x in centers[1:]:
            x_o = stage_coord[0] + y - h / 2.0
            y_o = stage_coord[1] + w / 2.0 - x
            z_o = stage_coord[2]
            #print x, y, ' ---> ', x_o, y_o, z_o
            tempStr = '"Cell%i", %f, %f, %f, 0, 0, FALSE, -9999, TRUE, TRUE, 0, -1, ""\n' % (i, x_o, y_o, z_o)
            fp.write(tempStr)
            i += 1
        fp.close()

        # write the original point (in order to get back to the origin)
        extension = os.path.splitext(filename)[-1]
        basefilename = os.path.splitext(filename)[0]
        origin_filename = basefilename + '_origin' + extension 
        fp = open(origin_filename, 'w')
        fp.write('"Stage Memory List", Version 6.0\n')
        fp.write('0, 0, 0, 0, 0, 0, 0, "um", "um"\n')
        fp.write('0\n')
        fp.write('1\n')
        tempStr = '"Cell1", %f, %f, %f, 0, 0, FALSE, -9999, TRUE, TRUE, 0, -1, ""\n' % (stage_coord[0], stage_coord[1], stage_coord[2])
        fp.write(tempStr)
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
        cell_labels, labels = self.find_neighbors(imbin, 100)

        if self.settings.debug:
            skimage.io.imsave(os.path.join(self.settings.img_debug_folder, 
                                           '%s16_basis_for_graph.png' % self.prefix), labels)
        
        # find the co-occurence matrix. 
        # The co-occurence matrix informs us about the neighboring relationships.
        # From there, we can build the graph.
        # attention : this does only work for < 256 objects. 
        # in order to fix this, I need to patch skimage. 
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
        imout = label_values[cell_labels]
        
        if self.settings.single_mask:
            skimage.io.imsave(os.path.join(self.settings.img_single_output_folder, 
                                           '%s_selected_cells.png' % self.prefix), imout)
            label_values[label_values==0] = 100
            label_values[0] = 0
            temp = label_values[cell_labels]
            skimage.io.imsave(os.path.join(self.settings.img_single_output_folder, 
                                           '%s_selected_and_other_cells.png' % self.prefix), temp)
                        
        return imout
    