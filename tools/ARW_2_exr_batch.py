#!/usr/bin/env python
#! -*- encoding: utf-8 -*-
import argparse
import glob
import subprocess
import os
import cv2
import time
from pathlib import Path
import subprocess
import multiprocessing
from multiprocessing import Pool

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

def treat(cmd):
    if not os.path.exists(str(exrFilename)):
        subprocess.call(cmd_str.split(' '))
        time.sleep(2)
        #Open And write with cv as halfFloat
        im = cv2.imread(str(exrFilename), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        cv2.imwrite(str(exrFilename),im,  [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])

cmd_list = []
for f in tqdm(files):
    print(f)
    assert Path(f).exists()
    exrFilename = Path('/home/ruizhu/Downloads/EXR') / (Path(f).stem + '.exr')
    cmd_str = 'darktable-cli %s %s %s --width 3000'%(f , args.xmp, str(exrFilename))
    cmd_list.append(cmd_str)

tic = time.time()
# print('==== executing %d commands...'%len(cmd_list))
# p = Pool(processes=opt.workers_total, initializer=init, initargs=(child_env,))
p = Pool(processes=8)

cmd_list = [(_cmd) for _cmd in enumerate(cmd_list)]
list(tqdm(p.imap_unordered(treat, cmd_list), total=len(cmd_list)))
p.close()
p.join()
print('==== ...DONE. Took %.2f seconds'%(time.time() - tic))


    # thrd = Thread(target=treat, args=(f, ))
    # thrd.start()
    # i+=1
    # if i%5 == 0:
    #     thrd.join()
    

    # if not Path(exrFilename).parent.exists():
    #     Path(exrFilename).parent.mkdir(parents=True, exist_ok=True)
        
    # if not os.path.exists(str(exrFilename)):
    #     cmd_str = 'darktable-cli %s %s %s --width 3000'%(f , args.xmp, str(exrFilename))
    #     print(cmd_str)
    #     subprocess.call(cmd_str.split(' '))
    #     time.sleep(2)
    #     assert Path(exrFilename).exists(), 'exr file not exist: %s'%exrFilename
    #     #Open And write with cv as halfFloat
    #     im = cv2.imread(str(exrFilename), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    #     cv2.imwrite(str(exrFilename),im,  [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
