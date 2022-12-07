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
    Blender Python API Documentation: https://docs.blender.org/api/current/index.html https://docs.blender.org/api/3.4/
    Intall Mitsuba addon for dumping Mitsuba scene into Blender: https://github.com/mitsuba-renderer/mitsuba-blender
        - First need to export the Mitsuba scene to **{XML file name}.blend** in Blender app: https://github.com/mitsuba-renderer/mitsuba-blender/wiki/Importing-a-Mitsuba-Scene
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
        assert self.os.pose_format in ['Blender', 'json'], 'support pose file for scaled Blender scene only!'

        self.blend_file = Path(str(self.os.xml_file).replace('.xml', '.blend'))
        assert self.blend_file.exists(), 'Blender file %s does not exist! See class documentation for export instructions.'%(self.blend_file.name)

        assert FORMAT == 'OPEN_EXR', 'only support this for now'

        bpy.ops.wm.open_mainfile(filepath=str(self.blend_file))
        scene = bpy.context.scene
        # Background
        scene.render.dither_intensity = 0.0
        scene.render.film_transparent = True
        scene.render.resolution_x = self.os.im_W_load
        scene.render.resolution_y = self.os.im_H_load
        scene.render.resolution_percentage = 100

        self.cam = scene.objects['Camera'] # the sensor in XML has to has 'id="Camera"'

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
        scene.view_layers["ViewLayer"].use_pass_potision = True

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
            'lighting_envmap', 
        ]

    def render(self):
        self.spp = self.im_params_dict.get('spp', 1024)

        if 'lighting_envmap' in self.modality_list:
            folder_name, render_folder_path = self.render_modality_check('lighting_envmap', force=True) # _: 'im', folder_name: 'Image'
            self.render_lighting_envmap(render_folder_path)
            return
            
        self.scene.render.resolution_x = self.os.im_W_load
        self.scene.render.resolution_y = self.os.im_H_load

        self.modal_file_outputs = []
        render_folder_path_list = []
        _modality_list = list(set(self.modality_list) - set(['lighting_envmap']))
        for _ in _modality_list:
            folder_name, render_folder_path = self.render_modality_check(_, force=True) # _: 'im', folder_name: 'Image'
            modal_file_output = self.tree.nodes.new(type="CompositorNodeOutputFile")
            modal_file_output.label = _
            self.links.new(self.render_layers.outputs[folder_name], modal_file_output.inputs[0]) # (self.render_layers.outputs[folder_name], bpy.data.scenes['Scene'].node_tree.nodes["File Output"].inputs[0])
            modal_file_output.base_path = str(render_folder_path)
            self.modal_file_outputs.append(modal_file_output)
            render_folder_path_list.append(render_folder_path)

        for _, render_folder_path in zip(_modality_list, render_folder_path_list):
                self.render_im(render_folder_path)
        
    def render_im(self, render_folder_path):
        print(blue_text('Rendering RGB to... by Mitsuba: %s')%str(render_folder_path))
        for i, (R_c2w_b, t_c2w_b) in enumerate(zip(self.os.R_c2w_b_list, self.os.t_c2w_b_list)):
            t_c2w_b = (t_c2w_b - self.os.trans_m2b) / self.os.scale_m2b # convert to Mitsuba scene scale (to match the dumped Blender scene from Mitsuba)
            # R_c2w_b = R_c2w_b @ np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]], dtype=np.float32)
            frame_id = self.os.frame_id_list[i]
            im_rendering_path = str(render_folder_path / ('%03d_0001'%frame_id))
            self.scene.render.filepath = str(im_rendering_path)
            self.cam.location = t_c2w_b.reshape(3, )
            euler_ = scipy.spatial.transform.Rotation.from_matrix(R_c2w_b).as_euler('xyz')
            assert np.abs(euler_[1]) < 1e-4, 'by default, no roll; otherwise something might be wrong loading the poses and converting to ruler angles'

            self.cam.rotation_euler[0] = euler_[0]
            self.cam.rotation_euler[1] = 0.
            assert self.cam.rotation_euler[1] < 1e-4, 'default camera has no roll'
            self.cam.rotation_euler[2] = euler_[2]

            self.cam.data.type = 'PERSP'
            
            for modal_file_output in self.modal_file_outputs:
                modal_file_output.file_slots[0].path = '%03d'%frame_id + '_'
            bpy.ops.render.render(write_still=True)  # render still

        print(blue_text('DONE.'))

    def render_lighting_envmap(self, render_folder_path):
        print(blue_text('Rendering lighting_envmap to... by Mitsuba: %s')%str(render_folder_path))
        lighting_global_xyz_list, lighting_global_pts_list = self.os.get_envmap_axes() # each of (env_row, env_col, 3, 3)
        env_row, env_col = self.os.lighting_params_dict['env_row'], self.os.lighting_params_dict['env_col']
        env_height, env_width = self.os.lighting_params_dict['env_height'], self.os.lighting_params_dict['env_width']
        self.scene.render.resolution_x = env_width
        self.scene.render.resolution_y = env_height
        # [panoramic-cameras] https://docs.blender.org/manual/en/latest/render/cycles/object_settings/cameras.html#panoramic-cameras
        self.cam.data.type = 'PANO'
        # [CyclesCameraSettings]https://docs.blender.org/api/2.80/bpy.types.CyclesCameraSettings.html#bpy.types.CyclesCameraSettings
        self.cam.data.cycles.panorama_type = 'EQUIRECTANGULAR'
        self.cam.data.cycles.latitude_min = 0.

        T_w_m2b = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], dtype=np.float32) # Mitsuba world to Blender world
        T_c_m2b = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=np.float32)

        for frame_id in tqdm(self.os.frame_id_list):
            lighting_global_xyz, lighting_global_pts = lighting_global_xyz_list[frame_id].reshape(-1, 3, 3), lighting_global_pts_list[frame_id].reshape(-1, 3, 3)
            assert lighting_global_xyz.shape[0] == env_row*env_col
            assert lighting_global_pts.shape[0] == env_row*env_col

            for env_idx, (xyz, pts) in tqdm(enumerate(zip(lighting_global_xyz, lighting_global_pts))):
                im_rendering_path = str(render_folder_path / ('%03d_%03d'%(frame_id, env_idx)))
                self.scene.render.filepath = str(im_rendering_path)
                
                pts_b = (T_w_m2b @ (pts.T)).T #  # Mitsuba -> Blender => xyz axes in blender coords
                self.cam.location = pts_b[0].reshape(3, ) # pts_b[0], pts_b[1], pts_b[2] should be the same

                at_vector_m = xyz[0]; up_m = xyz[2] # follow OpenRooms local hemisphere camera: images/openrooms_hemisphere.jpeg
                R_m = np.stack((np.cross(-up_m, at_vector_m), -up_m, at_vector_m), -1)
                assert np.abs(np.linalg.det(R_m)-1) < 1e-5
                R_b = T_w_m2b @ R_m @ T_c_m2b # Mitsuba -> Blender
                euler_ = scipy.spatial.transform.Rotation.from_matrix(R_b).as_euler('xyz')
                self.cam.rotation_euler[0] = euler_[0]
                self.cam.rotation_euler[1] = euler_[1]
                self.cam.rotation_euler[2] = euler_[2]

                bpy.ops.render.render(write_still=True)  # render still

        print(blue_text('DONE.'))

