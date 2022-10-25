import sys

# device = 'mm1'
device = 'mbp'
PATH_HOME = {
    'mbp': '/Users/jerrypiglet/Documents/Projects/OpenRooms_RAW_loader', 
    'mm1': '/home/ruizhu/Documents/Projects/OpenRooms_RAW_loader', 
    'qc': '/usr2/rzh/Documents/Projects/directvoxgorui'
}[device]
OR_RAW_ROOT = {
    'mbp': '/Users/jerrypiglet/Documents/Projects/data', 
    'mm1': '/newfoundland2/ruizhu/siggraphasia20dataset', 
    'qc': ''
}[device]

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

from lib.utils_misc import str2bool
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--vis_3d_plt', type=str2bool, nargs='?', const=True, default=False, help='whether to visualize 3D with plt for debugging')
parser.add_argument('--vis_3d_o3d', type=str2bool, nargs='?', const=True, default=True, help='whether to render in open3D')
parser.add_argument('--vis_2d_proj', type=str2bool, nargs='?', const=True, default=False, help='whether to show projection onto one image with plt (e.g. layout, object bboxes')
parser.add_argument('--if_shader', type=str2bool, nargs='?', const=True, default=False, help='')
parser.add_argument('--pcd_color_mode', type=str, default='rgb', help='if create color map for all points')
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

meta_split = 'main_xml1'
scene_name = 'scene0552_00_more'
frame_ids = [0, 1, 2, 3, 4] + list(range(5, 87, 10))
# frame_ids = [0]

openrooms_scene = openroomsScene3D(
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'rendering_root': base_root, 'xml_scene_root': xml_root, 'semantic_labels_root': semantic_labels_root, 'shape_pickles_root': shape_pickles_root, 
        'layout_root': layout_root, 'shapes_root': shapes_root, 'envmaps_root': envmaps_root}, 
    scene_params_dict={'meta_split': meta_split, 'scene_name': scene_name, 'frame_id_list': frame_ids}, 
    # modality_list = ['im_sdr', 'im_hdr', 'seg', 'poses', 'albedo', 'roughness', 'depth', 'normal', 'lighting_SG', 'lighting_envmap'], 
    modality_list = [
        'im_sdr', 'poses', 'seg', 
        'depth', 'normal', 
        # 'lighting_SG', 
        # 'lighting_envmap', 
        'layout', 
        'shapes', # objs + emitters, geometry shapes + emitter properties
        'mi', # mitsuba scene, loading from scene xml file
        ], 
    im_params_dict={
        'im_H_load': 480, 'im_W_load': 640, 'im_H_resize': 240, 'im_W_resize': 320
        }, 
    shape_params_dict={
        'if_load_obj_mesh': False, # set to False to not load meshes for objs (furniture) to save time
        'if_load_emitter_mesh': True,  # default True: to load emitter meshes, because not too many emitters
        },
    emitter_params_dict={
        'N_ambient_rep': '3SG-SkyGrd'
        },
    mi_params_dict={
        'if_also_dump_lit_lamps': True,  # True: to dump a second file containing lit-up lamps only
        'debug_dump_mesh': True, # [DEBUG] True: to dump all object meshes to mitsuba/meshes_dump; load all .ply files into MeshLab to view the entire scene: images/demo_mitsuba_dump_meshes.png
        'debug_render_test_image': False, # [DEBUG][slow] True: to render an image with first camera, usig Mitsuba: images/demo_mitsuba_render.png
        'if_sample_rays_pts': True, # True: to sample camera rays and intersection pts given input mesh and camera poses
        'if_get_segs': True, # True: to generate segs similar to those in openroomsScene2D.load_seg()
        },
)


if opt.vis_2d_proj:
    visualizer_2D = visualizer_openroomsScene_2D(
        openrooms_scene, 
        modality_list_vis=[
            'layout', 
            # 'shapes', 
            ], 
        frame_idx_list=[0, 1, 2, 3, 4], 
    )
    visualizer_2D.vis_2d_with_plt()

if opt.vis_3d_plt:
    visualizer_3D_plt = visualizer_openroomsScene_3D_plt(
        openrooms_scene, 
        modality_list_vis = [
            'layout', 
            'shapes', # boxes and labels (no meshes in plt visualization)
            'emitters', # emitter properties
            'emitter_envs', # emitter envmaps for (1) global envmap (2) half envmap & SG envmap of each window
            'mi_depth_normal', # compare depth & normal maps from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_depth_normals_2D.png
            'mi_seg', # compare segs from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_seg_2D.png
            ], 
    )
    visualizer_3D_plt.vis_3d_with_plt()


if opt.vis_3d_o3d:
    visualizer_3D_o3d = visualizer_openroomsScene_3D_o3d(
        openrooms_scene, 
        modality_list_vis=[
            # 'dense_geo', 
            'cameras', 
            # 'lighting_SG', 
            'layout', 
            'shapes', # bbox and (if loaded) meshs of shapes (objs + emitters)
            'emitters', # emitter properties (e.g. SGs, half envmaps)
            'mi', #mitsuba sampled rays, pts
            ], 
    )

    visualizer_3D_o3d.run_o3d(
        if_shader=opt.if_shader, # set to False to disable faycny shaders 
        cam_params={}, 
        dense_geo_params={
            'subsample_pcd_rate': 1, # change this according to how sparse the points you would like to be (also according to num of frame_ids)
            'if_ceiling': False, # [OPTIONAL] remove ceiling points to better see the furniture 
            'if_walls': False, # [OPTIONAL] remove wall points to better see the furniture 
            'if_normal': False, # [OPTIONAL] turn off normals to avoid clusters
            'subsample_normal_rate_x': 2, 
            'pcd_color_mode': opt.pcd_color_mode, 
            }, 
        lighting_SG_params={
            'subsample_lighting_SG_rate': 200, # change this according to how sparse the lighting arrows you would like to be (also according to num of frame_ids)
            # 'SG_scale': 1., 
            # 'SG_keep_ratio': 0.05, 
            # 'SG_clip_ratio': 0.1, 
            'SG_scale': 0.5, 
            'SG_keep_ratio': 0.2, 
            'SG_clip_ratio': 0.3, 
            'SG_autoscale': True, 
            }, 
        shapes_params={
            'simply_ratio': 0.1, # simply num of triangles to #triangles * simply_ratio
            'if_meshes': True, # [OPTIONAL] if show meshes for objs + emitters (False: only show bboxes)
            'if_labels': False, # [OPTIONAL] if show labels (False: only show bboxes)
        },
        emitters_params={
            'if_half_envmap': False, # [OPTIONAL] if show half envmap as a hemisphere for window emitters (False: only show bboxes)
            'scale_SG_length': 2., 
        },
        mi_params={
            'if_pts': True, # if show pts sampled by mi; should close to backprojected pts from OptixRenderer depth maps
            'if_pts_colorize_rgb': True, 
            'pts_subsample': 1,

            'if_cam_rays': False, 
            'cam_rays_if_pts': True, # if cam rays end in surface intersections; set to False to visualize rays of unit length
            'cam_rays_subsample': 10, 
            
            'if_normal': False, 
            'normal_subsample': 50, 
            'normal_scale': 0.2, 
        }, 
    )

# dump_path = Path(PATH_HOME) / ('logs/pickles/OR_public_re_gt_%s_#MOD_openrooms.pickle'%scene_name[5:])
# OR.fuse_3D_geometry(dump_path=dump_path)

# dump_path = Path(PATH_HOME) / ('logs/pickles/OR_public_re_gt_%s_#MOD_openrooms.pickle'%scene_name[5:])
# OR.fuse_3D_geometry(dump_path=dump_path)
