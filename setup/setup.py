"""
This is a setup.py script 

Usage:
    python setup.py py2exe
"""

#from setuptools import setup
from distutils.core import setup
import py2exe
import matplotlib
import skimage

import sys, os
sys.path.append("C:\\Users\\Thomas\\src\\SelectiveIllumination\\pysrc")

# manifest_template = '''
# <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
# <assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
# <assemblyIdentity
#     version="5.0.0.0"
#     processorArchitecture="x86"
#     name="%(prog)s"
#     type="win32"
# />
# <description>%(prog)s Program</description>
# <dependency>
#     <dependentAssembly>
#         <assemblyIdentity
#             type="win32"
#             name="Microsoft.Windows.Common-Controls"
#             version="6.0.0.0"
#             processorArchitecture="X86"
#             publicKeyToken="6595b64144ccf1df"
#             language="*"
#         />
#     <dependentAssembly>
#         <assemblyIdentity
#             type="win32"
#             name="Microsoft.VC90.CRT"
#             version="9.0.30729.4918"
#             processorArchitecture="X86"
#             publicKeyToken="1fc8b3b9a1e18e3b"
#             language="*"
#         />
#     </dependentAssembly>
# </dependency>
# </assembly>
# '''

DATA_FILES = []
OPTIONS = {#'argv_emulation': True,
           #'iconfile_resources' : 'SelectivIllu.icns',
           'packages': ['skimage', 'numpy', 'scipy'],
           'excludes': ['_gtkagg', '_tkagg'],
#           'includes': ['scipy.sparse.csgraph._validation'],
           'dll_excludes': ['libgdk-win32-2.0-0.dll', 'libgobject-2.0-0.dll'],           
           #'bundle_files': 1,
           }

folder = skimage.data_dir.replace('\\lib\\', '\\Lib\\')

skimage_data = [('skimage\\data', 
                 [os.path.join(folder, x) 
                  for x in os.listdir(folder)
                  if os.path.isfile(x)] )]


setup(
    windows = [
        {
            "script": "C:\\Users\\Thomas\\src\\SelectiveIllumination\\pysrc\\gui\\main_window.py",
            "icon_resources": [(1, "SelectiveIllu.ico")],
        }
    ],
    data_files=matplotlib.get_py2exe_datafiles() + skimage_data,
    options={'py2exe': OPTIONS,
             },
)

##setup(
##    console=APP,
##    data_files=DATA_FILES,
##    options={'py2exe': OPTIONS}
##)

