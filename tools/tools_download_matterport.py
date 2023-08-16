'''
python3 tools_download_matterport.py --output_dir /newfoundland3/ruizhu/Matterport3D/ --workers_total 16 --download (or --unzip)

The script download_mp.py works with Python2. Make sure you have an available Python2 binary locally.
'''
import multiprocessing
from multiprocessing import Pool
from pathlib import Path
import time

from tqdm import tqdm

house_scans_id_file = 'house_scans_id.txt'
assert Path(house_scans_id_file).exists()
types_str = 'cameras matterport_camera_intrinsics matterport_camera_poses matterport_color_images matterport_depth_images matterport_hdr_images matterport_mesh undistorted_camera_parameters undistorted_color_images undistorted_depth_images undistorted_normal_images house_segmentations region_segmentations'


house_scans_id = Path(house_scans_id_file).read_text().splitlines()
house_scans_id = [x.strip() for x in house_scans_id]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--workers_total', type=int, default=8, help='total num of workers; must be dividable by gpu_total, i.e. workers_total/gpu_total jobs per GPU')
parser.add_argument('--debug', action='store_true', help='not rendering; just showing missing files')
parser.add_argument('--output_dir', type=str, default='/newfoundland3/ruizhu/Matterport3D/', help='output directory')
parser.add_argument('--python_bin', type=str, default='python2', help='python binary')
parser.add_argument('--download', action='store_true', help='')
parser.add_argument('--unzip', action='store_true', help='')
parser.add_argument('--delete', action='store_true', help='')
parser.add_argument('--scenes', nargs='+', help='list of scene names', required=False, default=[])
# output_dir = '/newfoundland3/ruizhu/Matterport3D/'
opt = parser.parse_args()

assert Path(opt.output_dir).exists()

def exec(_):
    cmd = _
    print(cmd,)
    if not opt.debug:
        subprocess.call(cmd.split(' '))

if __name__ == '__main__':
    if opt.scenes != []:
        assert [_ in house_scans_id for _ in opt.scenes]
        house_scans_id = opt.scenes

    if opt.download:
        cmd = '%s download_mp.py -o %s --type %s --id %s'
        cmd_list = [cmd % (opt.python_bin, opt.output_dir, types_str, house_scan_id) for house_scan_id in house_scans_id]

    if opt.unzip:
        cmd_list = []
        scene_paths = [Path(opt.output_dir) / ('v1/scans/%s'%scene) for scene in house_scans_id]
        for scene_path in scene_paths:
            for types in types_str.split(' '):
                zip_path = scene_path / (types+'.zip')
                if zip_path.exists():
                    cmd = 'unzip -o %s -d %s'%(str(zip_path), str(Path(opt.output_dir) / 'v1' / 'scans'))
                    cmd_list.append(cmd)
                else:
                    print('WARNING: %s does not exist'%str(zip_path))

    assert opt.download or opt.unzip
    tic = time.time()
    p = Pool(processes=opt.workers_total)
    list(tqdm(p.imap_unordered(exec, cmd_list), total=len(cmd_list)))
    p.close()
    p.join()
    print('==== ...DONE. Took %.2f seconds'%(time.time() - tic))