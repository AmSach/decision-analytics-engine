# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['run.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'torchvision', 'torchaudio',
        'tensorflow', 'tensorboard', 'keras',
        'transformers', 'huggingface_hub', 'datasets',
        'cv2', 'PIL.TkImagePlugin', 'tkinter', '_tkinter',
        'IPython', 'ipykernel', 'jupyter', 'notebook', 'nbformat',
        'zmq', 'tornado', 'sqlalchemy', 'pytest', '_pytest',
        'onnxruntime', 'dask', 'numba', 'llvmlite',
        'opentelemetry', 'uvicorn', 'websockets', 'fastapi',
        'botocore', 'boto3', 'google', 'grpc', 'lxml',
        'spacy', 'nltk', 'scikit_learn', 'sklearn'
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DecisionGraph',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DecisionGraph',
)
