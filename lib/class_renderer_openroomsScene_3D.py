import numpy as np
from tqdm import tqdm

import mitsuba as mi
from lib.global_vars import mi_variant
mi.set_variant(mi_variant)

from lib.class_openroomsScene2D import openroomsScene2D
from lib.class_openroomsScene3D import openroomsScene3D

from lib.utils_PhySG import prepend_dims, hemisphere_int, lambda_trick

class renderer_openroomsScene_3D(object):
    '''
    A class for differentiable renderers of OpenRooms (public/public-re versions) scene contents.

    renderer options:
    - Zhengqin's renderer (Li et al., 2020, Inverse Rendering for Complex Indoor Scenes:)
        - input: per-pixel lighting envmap (e.g. 8x16); or SGs (to convert to envmaps)
    - PhySG renderer (Zhang et al., 2021, PhySG)
        - input: per-pixel lighting SGs (without having to convert to envmaps)
    '''
    def __init__(
        self, 
        openrooms_scene, 
        renderer_option: str, 
        pts_from: str='mi', # 'mi': ray-intersection with mitsuba scene; 'depth': backprojected from OptixRenderer renderer depth maps
    ):

        assert type(openrooms_scene) in [openroomsScene2D, openroomsScene3D], '[visualizer_openroomsScene] has to take an object of openroomsScene or openroomsScene3D!'
        self.os = openrooms_scene

        self.renderer_option = renderer_option
        assert self.renderer_option in ['ZQ', 'PhySG']

        self.pts_from = pts_from
        assert self.pts_from in ['mi', 'depth']
        if self.pts_from == 'mi':
            assert self.os.if_has_mitsuba_scene and self.os.pts_from['mi']
        if self.pts_from == 'depth':
            assert self.os.if_has_dense_geo and self.os.pts_from['depth']

    def render(self, frame_idx: int):
        '''
        frame_idx: 0-based indexing into all frames: [0, 1, ..., self.os.frame_num-1]
        '''
        if self.renderer_option == 'PhySG':
            self.render_PhySG(frame_idx)
        if self.renderer_option == 'ZQ':
            self.render_ZQ(frame_idx)

    def render_PhySG(self, frame_idx):
        '''
        Mostly adapted from https://github.com/Jerrypiglet/PhySG/blob/master/code/model/sg_render.py#L137
        '''

        assert self.os.if_has_lighting_SG
        assert self.os.if_load_lighting_SG_if_axis_global, 'need to convert to global SG axis first!'
        lighting_SG = self.os.lighting_SG_list[frame_idx] # (H, W, 12, 3+1+3)


    def render_ZQ(self, frame_idx):
        pass