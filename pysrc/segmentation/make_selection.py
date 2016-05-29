import os, re, time, sys

import numpy as np

from optparse import OptionParser

import skimage
import skimage.io
    
from settings import *

import segmentation.basic
import segmentation.selection
from lxml.html.builder import IMG


class Analyzer(object):

    def __init__(self, settings_filename=None, settings=None):
        if settings is None and settings_filename is None:
            raise ValueError("Either a settings object or a settings filename has to be given.")
        if not settings is None:
            self.settings = settings
        elif not settings_filename is None:
            self.settings = Settings(os.path.abspath(settings_filename), dctGlobals=globals())
            
        for folder in self.settings.make_folder:
            print folder
            if not os.path.isdir(folder):
                print 'made folder %s' % folder
                os.makedirs(folder)                        
        

    def process_folder(self, in_folder=None):
        if in_folder is None:
            in_folder = self.settings.data_folder
            
        filenames = filter(lambda x: os.path.splitext(x)[-1].lower() in ['.tiff', '.tif', '.png'], 
                           os.listdir(in_folder))
        for filename in filenames:
            self._process_single_file(os.path.join(in_folder, filename))            
        return

    
    def _process_single_file(self, filename):

        # read max projections
        # img = skimage.io.imread(filename)
        # imin = img[1,:,:]

        # read original image
        img = skimage.io.imread(filename)

        if len(img.shape) > 2:
            t1 = np.max(img, axis=3)
        
            # make max projection
            imin = np.max(t1, axis=0)
        else:
            imin = img
        sw = segmentation.basic.SimpleWorkflow(settings=self.settings, 
                                               prefix=os.path.splitext(os.path.basename(filename))[0].replace(' ', '_') + '__')
        
        # run segmentation algorithm
        res = sw(imin)

        # selection class        
        sel = segmentation.selection.Select(settings=self.settings, 
                                            prefix=os.path.splitext(os.path.basename(filename))[0].replace(' ', '_') + '__')

        # run selection
        cluster_dist = self.settings.cluster_dist
        max_extension = self.settings.max_extension        
        selected_cells = sel(imin, res, cluster_dist, max_extension=max_extension)
        
        # get the header information
        x, y, z = sel.read_header(filename)
        
        # write to text file
        if self.settings.coordinate_file:
            out_filename = os.path.join(self.settings.coordinate_folder, 'centers_px_%s.txt' % os.path.splitext(os.path.basename(filename))[0].replace(' ', '_'))
            sel.centers_to_px_text_file(selected_cells, out_filename)

            #out_filename = os.path.join(self.settings.coordinate_folder, 'centers_%s.txt' % os.path.splitext(os.path.basename(filename))[0].replace(' ', '_'))
            #sel.centers_to_text_file(selected_cells, out_filename)
        
        # write to metamorph
        if self.settings.metamorph_export:
            out_filename = os.path.join(self.settings.metamorph_folder, 'metamorph_%s.stg' % os.path.splitext(os.path.basename(filename))[0].replace(' ', '_'))
            sel.export_metamorph(selected_cells, out_filename, stage_coord=(x, y, z))
            
        return
    
        
        
# Windows: set PYTHONPATH=%PYTHONPATH%;C:\My_python_lib
# Windows: echo %PATH%

if __name__ ==  "__main__":

    description =\
'''
%prog - running segmentation tool .
'''

    parser = OptionParser(usage="usage: %prog [options]",
                         description=description)

    parser.add_option("-i", "--input_file", dest="input_file",
                      help="input file")
    parser.add_option("-o", "--output_folder", dest="output_folder",
                      help="output folder")
    parser.add_option("--input_folder", dest="input_folder",
                      help="input folder")
    parser.add_option("-s", "--settings_file", dest="settings_file",
                      help="Filename of the settings file")

    (options, args) = parser.parse_args()

    if (options.settings_file is None) or (options.output_folder is None):
        parser.error("incorrect number of arguments!")
    
    settings = Settings(os.path.abspath(options.settings_file), dctGlobals=globals())
    
    if (not options.output_folder is None):
        settings.result_folder = options.output_folder

    if (not options.input_folder is None):
        settings.data_folder = options.input_folder

    ana = Analyzer(settings=settings)
        
    if (not options.input_file is None):
        ana.process_single_file(options.input_file)
    else:
        ana.process_folder()
        
    
    