import os, re, time, sys
import shutil
import types

import numpy as np

from optparse import OptionParser

import skimage
import skimage.io
    
from settings import *

import segmentation.basic

import pdb

class Analyzer(object):

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

    def process_folder(self, in_folder, out_folder):
        filenames = filter(lambda x: os.path.splitext(x)[-1].lower() in ['.tiff', '.tif', '.png'], 
                           os.listdir(in_folder))
        for filename in filenames:
            self.process_single_file(os.path.join(in_folder, filename), out_folder)
            
        return

    def process_single_file(self, filename, output_folder):
        img_filename = os.path.basename(filename)
        prefix = os.path.splitext(img_filename)[0]
        output_file = os.path.join(output_folder, 'circle_centers_%s.txt' % prefix.replace(' ', '_'))
        self._process_single_file(filename, output_file)
        
        return
    
    def _process_single_file(self, filename, out_filename):

        # read max projections
        # img = skimage.io.imread(filename)
        # imin = img[1,:,:]

        # read original image
        img = skimage.io.imread(filename)
        t1 = img[:,1,:,:]

        # make max projection
        imin = np.max(t1, axis=0)        
        sw = segmentation.basic.SimpleWorkflow(settings=self.settings, 
                                               prefix=os.path.splitext(os.path.basename(filename))[0].replace(' ', '_') + '__')
        
        # run segmentation algorithm
        res = sw(imin)

        # selection class        
        sel = segmentation.basic.Select(settings=self.settings, 
                                        prefix=os.path.splitext(os.path.basename(filename))[0].replace(' ', '_') + '__')

        # run selection
        selected_cells = sel(imin, res, 1)
        
        # write file
        sel.centers_to_text_file(selected_cells, out_filename)
        
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

    ana = Analyzer(options.settings_file)
    if options.input_folder is None:
        if (options.input_file is None):
            parser.error("incorrect number of arguments!")
        ana.process_single_file(options.input_file, options.output_folder)
    else:
        ana.process_folder(options.input_folder, options.output_folder)
        
    
    