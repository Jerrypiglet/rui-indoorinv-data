#!/usr/bin/env python
#! -*- encoding: utf-8 -*-
import argparse
import glob
import subprocess
import os
import cv2
import time
from pathlib import Path
from threading import Thread

from tqdm import tqdm
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
parser = argparse.ArgumentParser()
parser.add_argument("folder")
parser.add_argument("xmp")
# parser.add_argument('--arw', action='store_true', 
#     help="Uses arw instead of cr2")
args = parser.parse_args()

# if args.arw:
assert Path(args.folder).exists()
files = glob.glob(args.folder+"/*.ARW")
# else:
# files = glob.glob(args.folder+"/*.CR2")

def treat(f):
    # exrFilename = (os.path.splitext(f)[0]+".exr").replace('\\', '/')
    # exrFilename = Path('/Volumes/RuiT7/ICCV23/real/IndoorKitchen_v2/EXR') / (Path(f).stem + '.exr')
    exrFilename = Path('/Users/jerrypiglet/Downloads/EXR') / (Path(f).stem + '.exr')

    if not Path(exrFilename).parent.exists():
        Path(exrFilename).parent.mkdir(parents=True, exist_ok=True)
        
    if not os.path.exists(str(exrFilename)):
        subprocess.call(['/Applications/darktable.app/Contents/MacOS/darktable-cli', f , args.xmp, str(exrFilename)])
        time.sleep(2)
        #Open And write with cv as halfFloat
        im = cv2.imread(str(exrFilename), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        cv2.imwrite(str(exrFilename),im,  [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])

i=0
for f in tqdm(files):
    print(f)
    assert Path(f).exists()
    thrd = Thread(target=treat, args=(f, ))
    thrd.start()
    i+=1
    if i%5 == 0:
        thrd.join()