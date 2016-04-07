"""
This is a setup.py script generated by py2applet

Usage:
    python setup.py py2app
"""

#from setuptools import setup
from distutils.core import setup
import py2exe

APP = ['../pysrc/gui/main_window.py']
DATA_FILES = []
OPTIONS = {#'argv_emulation': True,
           'iconfile' : 'SelectivIllu.icns',
           'packages': ['skimage', 'numpy', 'scipy']}

setup(
    console=APP,
    data_files=DATA_FILES,
    options={'py2exe': OPTIONS}
#    setup_requires=['py2exe'],
)
