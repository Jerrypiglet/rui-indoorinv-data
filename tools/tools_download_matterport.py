import subprocess
import multiprocessing
from multiprocessing import Pool
from pathlib import Path
import time

from tqdm import tqdm

python_bin  = 'python2'
house_scans_id_file = 'house_scans_id.txt'
assert Path(house_scans_id_file).exists()
types_str = 'cameras matterport_camera_intrinsics matterport_camera_poses matterport_color_images matterport_depth_images matterport_hdr_images matterport_mesh undistorted_camera_parameters undistorted_color_images undistorted_depth_images undistorted_normal_images house_segmentations region_segmentations'
assert Path(output_dir).exists()

cmd = 'python2 download_mp.py -o %s --type %s --id %s'

house_scans_id = Path(house_scans_id_file).read_text().splitlines()
house_scans_id = [x.strip() for x in house_scans_id]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--workers_total', type=int, default=8, help='total num of workers; must be dividable by gpu_total, i.e. workers_total/gpu_total jobs per GPU')
parser.add_argument('--debug', action='store_true', help='not rendering; just showing missing files')
parser.add_argument('--output_dir', type=str, default='/newfoundland3/ruizhu/Matterport3D/', help='output directory')
# output_dir = '/newfoundland3/ruizhu/Matterport3D/'

opt = parser.parse_args()

def download(_):
    cmd = _
    print(cmd,)
    if not opt.debug:
        subprocess.call(cmd.split(' '))

tic = time.time()
p = Pool(processes=opt.workers_total)
cmd_list = [cmd % (opt.output_dir, types_str, house_scan_id) for house_scan_id in house_scans_id]
list(tqdm(p.imap_unordered(download, cmd_list), total=len(cmd_list)))
p.close()
p.join()
print('==== ...DONE. Took %.2f seconds'%(time.time() - tic))



