
# host = 'mm1'
host = 'apple'
PATH_HOME = {
    'apple': '/Users/jerrypiglet/Documents/Projects/OpenRooms_RAW_loader', 
    'mm1': '/home/ruizhu/Documents/Projects/OpenRooms_RAW_loader', 
    'qc': '/usr2/rzh/Documents/Projects/directvoxgorui', 
}[host]
OR_RAW_ROOT = {
    'apple': '/Users/jerrypiglet/Documents/Projects/data', 
    'mm1': '/newfoundland2/ruizhu/siggraphasia20dataset', 
    'qc': '', 
}[host]

import sys
sys.path.insert(0, PATH_HOME)
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
np.set_printoptions(suppress=True)
from lib.utils_io import load_matrix, load_img, load_binary, load_h5

from lib.class_openroomsScene3D import openroomsScene3D
from lib.class_visualizer_openroomsScene_2D import visualizer_openroomsScene_2D
from lib.class_visualizer_openroomsScene_3D_o3d import visualizer_openroomsScene_3D_o3d
from lib.class_visualizer_openroomsScene_3D_plt import visualizer_openroomsScene_3D_plt
from lib.class_renderer_openroomsScene_3D import renderer_openroomsScene_3D

from lib.utils_misc import str2bool
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--vis_3d_plt', type=str2bool, nargs='?', const=True, default=False, help='whether to visualize 3D with plt for debugging')
parser.add_argument('--vis_3d_o3d', type=str2bool, nargs='?', const=True, default=True, help='whether to visualize in open3D')
parser.add_argument('--vis_2d_proj', type=str2bool, nargs='?', const=True, default=False, help='whether to show projection onto one image with plt (e.g. layout, object bboxes')
parser.add_argument('--if_shader', type=str2bool, nargs='?', const=True, default=False, help='')
parser.add_argument('--pcd_color_mode', type=str, default='rgb', help='if create color map for all points')
parser.add_argument('--render_3d', type=str2bool, nargs='?', const=True, default=False, help='differentiable surface rendering')
parser.add_argument('--renderer_option', type=str, default='PhySG', help='differentiable renderer option')
opt = parser.parse_args()

base_root = Path(PATH_HOME) / 'data/public_re_3'
xml_root = Path(PATH_HOME) / 'data/public_re_3/scenes'
# intrinsics_path = Path(PATH_HOME) / 'data/intrinsic.txt'
semantic_labels_root = Path(PATH_HOME) / 'files_openrooms'
layout_root = Path(OR_RAW_ROOT) / 'layoutMesh'
shapes_root = Path(OR_RAW_ROOT) / 'uv_mapped'
envmaps_root = Path(OR_RAW_ROOT) / 'EnvDataset' # not publicly availale
shape_pickles_root = Path(PATH_HOME) / 'data/openrooms_shape_pickles' # for caching shape bboxes so that we do not need to load meshes very time if only bboxes are wanted
if not shape_pickles_root.exists():
    shape_pickles_root.mkdir()

'''
The classroom scene: one lamp (lit up) + one window (less sun)
data/public_re_3/main_xml1/scene0552_00_more/im_4.png
'''
# meta_split = 'main_xml1'
# scene_name = 'scene0552_00_more'
frame_ids = list(range(87))

'''
The classroom scene: one lamp (dark) + one window (directional sun)
data/public_re_3/mainDiffLight_xml1/scene0552_00_more/im_4.png
'''
meta_split = 'mainDiffLight_xml1'
scene_name = 'scene0552_00_more'
frame_ids = list(range(87))

'''
The lounge with very specular floor and 3 lamps
data/public_re_3/main_xml/scene0008_00_more/im_58.png
'''
# meta_split = 'main_xml'
# scene_name = 'scene0008_00_more'
# frame_ids = list(range(102))

'''
The conference room with a ceiling lamp; shiny chairs and floor
data/public_re_3/main_xml/scene0005_00_more/im_58.png
'''
# meta_split = 'main_xml'
# scene_name = 'scene0005_00_more'
# frame_ids = list(range(102))

openrooms_scene = openroomsScene3D(
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'rendering_root': base_root, 'xml_scene_root': xml_root, 'semantic_labels_root': semantic_labels_root, 'shape_pickles_root': shape_pickles_root, 
        'layout_root': layout_root, 'shapes_root': shapes_root, 'envmaps_root': envmaps_root}, 
    scene_params_dict={'meta_split': meta_split, 'scene_name': scene_name, 'frame_id_list': frame_ids}, 
    # modality_list = ['im_sdr', 'im_hdr', 'seg', 'poses', 'albedo', 'roughness', 'depth', 'normal', 'lighting_SG', 'lighting_envmap'], 
    modality_list = [
        'im_sdr', 'poses', 'seg', 
        'im_hdr', 'albedo', 'roughness', 
        'depth', 'normal', 
        # 'lighting_SG', 
        # 'lighting_envmap', 
        'layout', 
        'shapes', # objs + emitters, geometry shapes + emitter properties
        'mi', # mitsuba scene, loading from scene xml file
        ], 
    im_params_dict={
        'im_H_load': 480, 'im_W_load': 640, 
        'im_H_resize': 240, 'im_W_resize': 320
        }, 
    shape_params_dict={
        'if_load_obj_mesh': True, # set to False to not load meshes for objs (furniture) to save time
        'if_load_emitter_mesh': True,  # default True: to load emitter meshes, because not too many emitters
        },
    mi_params_dict={
        'if_also_dump_lit_lamps': True,  # True: to dump a second file containing lit-up lamps only
        'debug_dump_mesh': True, # [DEBUG] True: to dump all object meshes to mitsuba/meshes_dump; load all .ply files into MeshLab to view the entire scene: images/demo_mitsuba_dump_meshes.png
        'debug_render_test_image': False, # [DEBUG][slow] True: to render an image with first camera, usig Mitsuba: images/demo_mitsuba_render.png
        'if_sample_rays_pts': True, # True: to sample camera rays and intersection pts given input mesh and camera poses
        'if_get_segs': True, # True: to generate segs similar to those in openroomsScene2D.load_seg()
        },
)

openrooms_scene.pose_list # [R|t]; camera coordinates is in OpenCV convention (right-down-forward) (same as ScanNet)
openrooms_scene.im_hdr_list # HDR images
openrooms_scene.K # intrinsics; again in OpenCV convention
openrooms_scene.im_H_resize, openrooms_scene.im_W_resize # original rendered images are of 480x640; resized to this dimension

'''
dump mesh
'''
import copy
from lib.utils_OR.utils_OR_mesh import writeMesh

num_vertices = 0
f_list = []
for vertices, faces in zip(openrooms_scene.vertices_list, openrooms_scene.faces_list):
    f_list.append(copy.deepcopy(faces + num_vertices))
    num_vertices += vertices.shape[0]
v_list = copy.deepcopy(openrooms_scene.vertices_list)
writeMesh('./tmp_mesh.obj', np.vstack(v_list), np.vstack(f_list))
