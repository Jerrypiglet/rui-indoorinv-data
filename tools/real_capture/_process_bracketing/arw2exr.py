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
parser.add_argument("output_path") # '/home/liwen/Documents/EXR'
args = parser.parse_args()

# if args.arw:
assert Path(args.folder).exists()
files = glob.glob(args.folder+"/*.ARW")

for f in tqdm(files):
    print(f)
    assert Path(f).exists()
    exrFilename = Path(args.output_path) / (Path(f).stem + '.exr')
    cmd_list = ['darktable-cli', f , args.xmp, str(exrFilename), '--width', '3000']
    print(' '.join(cmd_list))
    subprocess.call(cmd_list)