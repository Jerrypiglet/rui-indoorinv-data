import sys
PATH_HOME = '/Users/jerrypiglet/Documents/Projects/OpenRooms_RAW_loader'
# PATH_HOME = '/home/ruizhu/Documents/Projects/dvgomm1'
# PATH_HOME = '/usr2/rzh/Documents/Projects/directvoxgorui'
sys.path.insert(0, PATH_HOME)
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
np.set_printoptions(suppress=True)
from lib.utils_io import load_matrix, load_img, load_binary, load_h5

from lib.class_openroomsScene2D import openroomsScene2D
from lib.class_visualizer_openroomsScene_2D import visualizer_openroomsScene_2D

from lib.utils_misc import str2bool
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--vis_2d', type=str2bool, nargs='?', const=True, default=True, help='whether to show projection onto one image with plt (e.g. layout, object bboxes')
parser.add_argument('--if_shader', type=str2bool, nargs='?', const=True, default=True, help='')
opt = parser.parse_args()

base_root = Path(PATH_HOME) / 'data/openrooms_public_re_2'
xml_root = Path(PATH_HOME) / 'data/openrooms_public_re_2/scenes'
intrinsics_path = Path(PATH_HOME) / 'data/intrinsic.txt'
semantic_labels_root = Path(PATH_HOME) / 'data'

# meta_split = 'main_xml'
# scene_name = 'scene0008_00'
# frame_ids = [58] + list(range(1, 102, 50)) # scene_name = 'scene0008_00' # every 10; scene0008_00 public_re

# scene_name = 'scene0008_00'
# frame_ids = [58] + list(range(1, 102, 50)) # scene_name = 'scene0008_00' # every 10; scene0008_00 public_re

# scene_name = 'scene0005_00'
# frame_ids = [15] + list(range(1, 113, 10))

meta_split = 'main_xml1'
scene_name = 'scene0552_00_more'
frame_ids = list(range(0, 87, 10))

openrooms_scene = openroomsScene2D(
    root_path_dict = {
        'PATH_HOME': Path(PATH_HOME), 
        'rendering_root': Path(base_root), 
        'xml_scene_root': Path(xml_root), 
        'semantic_labels_root': semantic_labels_root, 
        'intrinsics_path': Path(intrinsics_path)
        }, 
    scene_params_dict={
        'meta_split': meta_split, 
        'scene_name': scene_name, 
        'frame_id_list': frame_ids}, 
    modality_list = [
        'im_sdr', 'im_hdr', 'seg', 'poses', 
        'albedo', 'roughness', 
        'depth', 'normal',
        'matseg', 'semseg', 
        # 'lighting_SG', 'lighting_envmap'
        ], 
    # modality_list = ['im_sdr', 'poses', 'depth', 'normal', 'lighting_SG'], 
    im_params_dict={'im_H_load': 480, 'im_W_load': 640, 'im_H_resize': 240, 'im_W_resize': 320}, 
)

if opt.vis_2d:
    vis_2D = visualizer_openroomsScene_2D(
        openrooms_scene, 
        modality_list=[
            'depth', 'normal', 'albedo', 'roughness', # images/demo_all_2D.png
            # 'seg_area', 'seg_env', 'seg_obj', 
            # 'matseg', # images/demo_semseg_matseg_2D.png
            # 'semseg', # images/demo_semseg_matseg_2D.png
            ], 
        frame_idx_list=[0, 1, 2, 3, 4], # 0-based indexing of all selected frames
    )
    vis_2D.vis_2d_with_plt()