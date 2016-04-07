# -*- mode: python -*-

block_cipher = None


a = Analysis(['..\\pysrc\\gui\\main_window.py'],
             pathex=['C:\\Users\\Thomas\\src\\SelectiveIllumination\\pysrc', 'C:\\Users\\Thomas\\src\\SelectiveIllumination\\setup'],
             binaries=None,
             datas=None,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='main_window',
          debug=True,
          strip=False,
          upx=True,
          console=True )
