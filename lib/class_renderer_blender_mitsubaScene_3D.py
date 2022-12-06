import shutil
import glob
from tqdm import tqdm
import numpy as np
np.set_printoptions(suppress=True)

from pathlib import Path
import bpy
import scipy
from lib.utils_misc import blue_text, yellow, red
from lib.utils_io import convert_write_png

from .class_rendererBase import rendererBase

class renderer_blender_mitsubaScene_3D(rendererBase):
    '''
    A class used to render Mitsuba scene with Blender.
    Build and install bpy package: https://wiki.blender.org/wiki/Building_Blender/Other/BlenderAsPyModule (by default, assume Python 3.10)
        - for Python 3.8 or other versions (but may not support features like Metal on Mac): 
            git checkout blender-v2.92-release && make update
            cmake -DPYTHON_VERSION=3.8 ../blender
    '''
    def __init__(
        self, 
        mitsuba_scene, 
        modality_list, 
        host, 
        FORMAT='OPEN_EXR', 
        COLOR_DEPTH=16, 
        *args, **kwargs, 
    ):
        rendererBase.__init__(
            self, 
            mitsuba_scene, 
            modality_list, 
            *args, **kwargs
        )

        assert FORMAT == 'OPEN_EXR', 'only support this for now'

        scene = bpy.context.scene
        # Background
        scene.render.dither_intensity = 0.0
        scene.render.film_transparent = True
        scene.render.resolution_x = self.os.im_W_load
        scene.render.resolution_y = self.os.im_H_load
        scene.render.resolution_percentage = 100

        self.cam = scene.objects['Camera']
        obj_idx = 1
        for obj in bpy.context.scene.objects:
            if obj.type in ('MESH'):
                obj.pass_index=obj_idx
                obj_idx += 1

        scene.render.image_settings.file_format = FORMAT
        scene.render.image_settings.color_depth = str(COLOR_DEPTH)

        # Set pass
        scene.view_layers["ViewLayer"].use_pass_normal = True
        scene.view_layers["ViewLayer"].use_pass_object_index = True
        scene.view_layers["ViewLayer"].use_pass_z = True
        scene.view_layers["ViewLayer"].use_pass_diffuse_color = True

        scene.use_nodes = True
        self.tree = scene.node_tree
        self.links = self.tree.links
        for n in self.tree.nodes:
            self.tree.nodes.remove(n)
            
        self.render_layers = self.tree.nodes.new('CompositorNodeRLayers')

        # Set up renderer params
        self.scene = bpy.data.scenes["Scene"]
        self.scene.render.engine = 'CYCLES'
        self.scene.render.use_motion_blur = False
        self.scene.cycles.device = {
            'apple': 'CPU', 
            'mm1': 'GPU', 
            'qc': 'GPU', 
        }[host]

        self.scene.render.film_transparent = True
        self.scene.view_layers[0].cycles.use_denoising = True

        self.spp = self.im_params_dict.get('spp', 128)
        self.scene.cycles.samples = self.spp

        cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
        # cycles_prefs.compute_device_type = 'CUDA'
        cycles_prefs.compute_device_type = {
            'apple': 'METAL', 
            'mm1': 'CUDA', 
            'qc': 'CUDA', 
        }[host]
        cycles_prefs.get_devices()
        for di, device in enumerate(cycles_prefs.devices):
            device.use = (di == 0)

        
    @property
    def valid_modalities(self):
        return [
            'im', 
            # 'seg', 
            'albedo', 'roughness', 
            'depth', 'normal', 
            # 'lighting_envmap', 
        ]

    def render(self):
        for _ in self.modality_list:
            folder_name, render_folder_path = self.render_modality_check(_)
            modal_file_output = self.tree.nodes.new(type="CompositorNodeOutputFile")
            modal_file_output.label = _
            self.links.new(self.render_layers.outputs[folder_name], modal_file_output.inputs[0])
            modal_file_output.base_path = str(render_folder_path)

            if _ == 'im': self.render_im(modal_file_output, render_folder_path)
        
    def render_im(self, modal_file_output, render_folder_path):
        self.spp = self.im_params_dict.get('spp', 1024)

        print(blue_text('Rendering RGB to... by Mitsuba: %s')%str(render_folder_path))
        for i, (R_c2w_b, t_c2w_b) in enumerate(zip(self.os.R_c2w_b_list, self.os.t_c2w_b_list)):
            im_rendering_path = str(render_folder_path / ('%03d_0001'%i))
            self.scene.render.filepath = str(im_rendering_path)
            self.cam.location = t_c2w_b.reshape(3, )
            euler_ = scipy.spatial.transform.Rotation.from_matrix(R_c2w_b).as_euler('xyz')
            # self.cam.rotation_euler[0] = euler_[0]
            # self.cam.rotation_euler[2] = euler_[2]
            modal_file_output.file_slots[0].path = '%03d'%i + '_'
            bpy.ops.render.render(write_still=True)  # render still

        print(blue_text('DONE.'))

