"""
This is a setup.py script 

Usage:
    python setup.py py2exe
"""

#from setuptools import setup
from distutils.core import setup
import py2exe

import sys
sys.path.append("C:\\Users\\Thomas\\src\\SelectiveIllumination\\pysrc")

DATA_FILES = []
OPTIONS = {#'argv_emulation': True,
           #'iconfile_resources' : 'SelectivIllu.icns',
           'packages': ['skimage', 'numpy', 'scipy']}

setup(
    windows = [
        {
            "script": "C:\\Users\\Thomas\\src\\SelectiveIllumination\\pysrc\\gui\\main_window.py",
            "icon_resources": [(1, "SelectiveIllu.icns")],
        }
    ],
    options={'py2exe': OPTIONS},
)

##setup(
##    console=APP,
##    data_files=DATA_FILES,
##    options={'py2exe': OPTIONS}
##)

