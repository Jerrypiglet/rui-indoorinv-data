from tqdm import tqdm
import numpy as np
np.set_printoptions(suppress=True)
import time

from pathlib import Path
import bpy
import scipy
import json

from lib.utils_misc import get_list_of_keys, white_blue, blue_text, blue_text, red, listify_matrix

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
        blender_file_name=None, 
        if_skip_check: bool=False,
        *args, **kwargs, 
    ):
        rendererBase.__init__(
            self, 
            mitsuba_scene=mitsuba_scene, 
            modality_list=modality_list, 
            if_skip_check=if_skip_check, 
            *args, **kwargs
        )
        # assert self.os.pose_format in ['Blender', 'json'], ''

        self.blend_file_path = Path(str(self.os.xml_file_path).replace('.xml', '.blend')) if blender_file_name is None else self.os.scene_path / blender_file_name
        assert self.blend_file_path.exists(), 'Blender file %s does not exist! See class documentation for export instructions.'%(self.blend_file_path.name)
        # if not self.blend_file_path.exists(): # 'Blender file %s does not exist! See class documentation for export instructions.'%(self.blend_file_path.name)
        #     bpy.ops.wm.save_as_mainfile(filepath=str(self.blend_file_path.absolute()))
        #     bpy.ops.wm.open_mainfile(filepath=str(self.blend_file_path.absolute()))
        #     # [TODO] this somehow does not work although the Mitsuba add-on is installed and manual import works: File -> Import -> Mitsuba (.xml)
        #     bpy.ops.import_scene.mitsuba(filepath=str(self.os.xml_file_path.absolute()), override_scene=True) # make sure you can find File -> Import -> Mitsuba (.xml) in the Blender app associated with bpy; otherwise install Mitsuba add-on for Blender
        #     bpy.ops.wm.save_mainfile()

        assert FORMAT == 'OPEN_EXR', 'only support this for now'
        self.spp = self.im_params_dict.get('spp', 128)
        
        bpy.ops.wm.open_mainfile(filepath=str(self.blend_file_path))
        self.scene = bpy.context.scene
        self.scene = bpy.data.scenes["Scene"]
        self.scene.render.use_motion_blur = False
        self.scene.view_layers[0].cycles.use_denoising = True
        # self.scene.view_layers[0].cycles.use_denoising = False
        self.scene.cycles.samples = self.spp
        
        # https://docs.blender.org/manual/en/latest/render/color_management.html
        self.scene.view_settings.view_transform = 'Standard'
        # self.scene.view_settings.view_transform = 'Raw'
        
        '''
        configure render engine and device
        '''
        
        print("----------------------------------------------")
        print('setting up gpu/metal ......')

        self.scene.render.engine = 'CYCLES'
        
        cycles_device = {
            'apple': 'CPU', 
            'mm1': 'GPU', 
            'r4090': 'GPU', 
        }[host]
        compute_device_type = {
            'apple': 'METAL', 
            'mm1': 'CUDA', 
            'r4090': 'CUDA', 
        }[host]
        bpy.context.scene.cycles.device = cycles_device
        # for scene in bpy.data.scenes:
        #     print(scene.name)
        #     scene.cycles.device = cycles_device
        if cycles_device == 'GPU':
            self.scene.cycles.denoiser = 'OPTIX'

        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = compute_device_type
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        print('==== compute_device_type: ', white_blue(bpy.context.preferences.addons["cycles"].preferences.compute_device_type))

        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            d.use = True
            if d.type == 'CPU':
                d.use = False
            print("Device '{}' type {} :".format(d.name, d.type), white_blue(str(d.use)) if d.use else red(str(d.use)))
        print('setting up gpu/metal done')
        print("----------------------------------------------")

        '''
        modalities: deal with aov modalities
        '''
        
        AOV_MODALS = []
        if 'roughness' in self.modality_list:
            AOV_MODALS.append('Roughness') #,'Metallic'
        # assigne material id
        for i, mat in enumerate(bpy.data.materials):
            mat.pass_index=(i+1)
        # link aov output
        for aov_modal in AOV_MODALS:
            for mat in bpy.data.materials:
                tree = mat.node_tree
                if tree is None:
                    continue
                for node in tree.nodes:
                    if 'Bsdf' not in node.bl_idname:
                        continue
                    if (node.bl_idname =='ShaderNodeBsdfGlossy' or node.bl_idname =='ShaderNodeBsdfGlass') and node.distribution == 'SHARP' and aov_modal=='Roughness':
                        modal_value = 0.0
                        buffer_node = tree.nodes.new('ShaderNodeValue')
                        from_socket = buffer_node.outputs['Value']
                        from_socket.default_value = modal_value
                    elif aov_modal not in node.inputs.keys():
                        modal_value = 0.0
                        buffer_node = tree.nodes.new('ShaderNodeValue')
                        from_socket = buffer_node.outputs['Value']
                        from_socket.default_value = modal_value
                    else:
                        socket = node.inputs[aov_modal]
                        if len(socket.links) == 0:
                            if node.bl_idname == 'ShaderNodeBsdfDiffuse' and aov_modal == 'Roughness':
                                modal_value = 1.0
                            else:
                                modal_value = socket.default_value
                            buffer_node = tree.nodes.new('ShaderNodeValue')
                            from_socket = buffer_node.outputs['Value']
                            from_socket.default_value = modal_value
                            tree.links.new(from_socket,socket)
                        else:
                            from_socket=socket.links[0].from_socket

                    aov_node = tree.nodes.new('ShaderNodeOutputAOV')
                    aov_node.name = aov_modal
                    tree.links.new(from_socket,aov_node.inputs['Value'])

        '''
        scene
        '''
        
        # Background
        self.scene.render.dither_intensity = 0.0
        self.scene.render.film_transparent = True
        self.scene.render.resolution_x = self.im_params_dict['im_W_load']
        self.scene.render.resolution_y = self.im_params_dict['im_H_load']
        self.scene.render.resolution_percentage = 100

        # self.cam = scene.objects['Camera'] # the sensor in XML has to has 'id="Camera"'
        self.cam = bpy.context.scene.camera #scene.objects['Camera'] # self.cam.data.lens -> 31.17691421508789, in mm (not degrees)
        self.cam.data.clip_start = 0.05
        self.cam.data.clip_end = 50.0
        # import ipdb; ipdb.set_trace()
        # print camera parameters
        w = self.scene.render.resolution_x
        h = self.scene.render.resolution_y
        cx = np.float32(-self.cam.data.shift_x + 0.5)
        cy = np.float32(self.cam.data.shift_y * w / h + 0.5)
        # self.cam.data.shift_x = 0.5
        # self.cam.data.shift_y = -0.5 / w * h
        
        # Assumes square pixels:
        # self.scene.render.pixel_aspect_x == self.scene.render.pixel_aspect_y
        fx = np.float32(self.cam.data.sensor_width / 2. / self.cam.data.lens)
        fx_K = self.os.K[0][2]/self.os.K[0][0] # should be the same as fx
        
        fov_x = np.arctan(fx)/np.pi*180.*2. # in degrees
        print('Blender camera parameters: w %d, h %d, cx %.4f, cy %.4f; fx %.4f, fx from K %.4f, fov_x: %.4f'%(w, h, cx, cy, fx, fx_K, fov_x))

        obj_idx = 1
        for obj in bpy.context.scene.objects:
            if obj.type in ('MESH'):
                obj.pass_index=obj_idx
                obj_idx += 1

        # self.scene.render.image_settings.file_format = FORMAT
        self.scene.render.image_settings.file_format = FORMAT
        self.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

        # Set pass
        # self.scene.view_layers["ViewLayer"].use_pass_normal = True # "ViewLayer" not found: https://zhuanlan.zhihu.com/p/533843765
        # self.scene.view_layers["ViewLayer"].use_pass_object_index = True
        # self.scene.view_layers["ViewLayer"].use_pass_z = True
        # self.scene.view_layers["ViewLayer"].use_pass_material_index = True
        # self.scene.view_layers["ViewLayer"].use_pass_diffuse_color = True
        # self.scene.view_layers["ViewLayer"].use_pass_emit = True
        # self.scene.view_layers["ViewLayer"].use_pass_glossy_color = True
        # self.scene.view_layers["ViewLayer"].use_pass_position = True
        
        self.scene.view_layers[0].use_pass_normal = True # "ViewLayer" not found: https://zhuanlan.zhihu.com/p/533843765
        self.scene.view_layers[0].use_pass_object_index = True
        self.scene.view_layers[0].use_pass_z = True
        self.scene.view_layers[0].use_pass_material_index = True
        self.scene.view_layers[0].use_pass_diffuse_color = True
        self.scene.view_layers[0].use_pass_glossy_color = True
        self.scene.view_layers[0].use_pass_emit = True
        self.scene.view_layers[0].use_pass_position = True

        self.scene.use_nodes = True

        for aov_modal in AOV_MODALS:
            bpy.ops.scene.view_layer_add_aov()
            # self.scene.view_layers["ViewLayer"].aovs[-1].name = aov_modal
            # self.scene.view_layers["ViewLayer"].aovs[-1].type = "VALUE"
            self.scene.view_layers[0].aovs[-1].name = aov_modal
            self.scene.view_layers[0].aovs[-1].type = "VALUE"

        # self.scene = bpy.data.scenes["Scene"]
        self.tree = self.scene.node_tree
        self.links = self.tree.links
        for n in self.tree.nodes:
            self.tree.nodes.remove(n)
        self.render_layers = self.tree.nodes.new('CompositorNodeRLayers')

    @property
    def valid_modalities(self):
        return [
            'im', 
            'albedo', 'roughness', 
            'depth', 'normal', 
            'index', 
            'emission', 
            'lighting_envmap', 
        ]

    def render(self, if_force: bool=False):
        '''
        render all modalities except for 'lighting_envmap'
        '''
        if 'lighting_envmap' in self.modality_list:
            env_height, env_width, env_row, env_col = get_list_of_keys(self.lighting_params_dict, ['env_height', 'env_width', 'env_row', 'env_col'], [int, int, int, int])
            folder_name_appendix = '-%dx%dx%dx%d'%(env_row, env_col, env_height, env_width)
            folder_name, render_folder_path = self.render_modality_check('lighting_envmap', folder_name_appendix=folder_name_appendix, if_force=if_force) # _: 'im', folder_name: 'Image'
            self.render_lighting_envmap(render_folder_path)
            return
            
        # npy_file_path = self.os.pose_file_root / ('%s.npy'%self.os.split); assert npy_file_path.exists(), npy_file_path
        # blender_poses = np.load(npy_file_path) # (N, 2, 3)
        from utils_OR.utils_OR_cam import convert_OR_poses_to_blender_npy
        blender_poses = convert_OR_poses_to_blender_npy(pose_list=self.os.pose_list)
        assert len(blender_poses) == self.os.frame_num
        
        # blender_poses = blender_poses[0:1] # DEBUG
        
        self.modal_file_outputs = []
        _modality_list = list(set(self.modality_list) - set(['lighting_envmap']))
        
        for modality in _modality_list:
            folder_name, render_folder_path = self.render_modality_check(modality, if_force=if_force) # _: 'im', folder_name: 'Image'
            modal_file_output = self.tree.nodes.new(type="CompositorNodeOutputFile")
            modal_file_output.label = folder_name
            self.links.new(self.render_layers.outputs[folder_name], modal_file_output.inputs[0]) # (self.render_layers.outputs[folder_name], bpy.data.scenes['Scene'].node_tree.nodes["File Output"].inputs[0])
            modal_file_output.base_path = str(render_folder_path)
            self.modal_file_outputs.append(modal_file_output)

        im_rendering_folder = self.scene_rendering_path / 'Image'
        
        print(blue_text('Rendering modalities for %d frames to... by Blender: %s')%(len(blender_poses), str(im_rendering_folder)))
        
        # Data to store in JSON file
        out_data = {
            'camera_angle_x': bpy.context.scene.camera.data.angle_x, #bpy.data.objects['Camera'].data.angle_x,
            'camera_angle_y': bpy.context.scene.camera.data.angle_y, #bpy.data.objects['Camera'].data.angle_y,
        }
        out_data['frames'] = []
        
        for i in range(len(blender_poses)):
            # t_c2w_b = (t_c2w_b - self.os.trans_m2b) / self.os.scale_m2b # convert to Mitsuba scene scale (to match the dumped Blender scene from Mitsuba)
            frame_id = self.os.frame_id_list[i]
            im_rendering_path = str(im_rendering_folder / ('%03d_0001'%frame_id))
            self.scene.render.filepath = str(im_rendering_path)
            
            self.cam.location = blender_poses[i, 0]
            self.cam.rotation_euler[0] = blender_poses[i, 1, 0]
            self.cam.rotation_euler[2] = blender_poses[i, 1, 2]
            self.cam.rotation_euler[1] = blender_poses[i, 1, 1]
            # self.cam.rotation_euler[1] = 0.
            # assert self.cam.rotation_euler[1] < 1e-4, 'default camera has no roll'
            
            self.cam.data.type = 'PERSP'
            
            # import ipdb; ipdb.set_trace()
            
            for modal_file_output in self.modal_file_outputs:
                modal_file_output.file_slots[0].path = '%03d'%frame_id + '_'
            bpy.ops.render.render(write_still=True)  # render still
            
            frame_data = {
                'file_path': self.scene.render.filepath,
                'transform_matrix': listify_matrix(self.cam.matrix_world)
            }
            out_data['frames'].append(frame_data)

        print(blue_text('render_modality. DONE.'))
        
        self.dump_transforms_json(out_data)
                
    def dump_transforms_json(self, out_data):
        json_file_path = self.os.pose_file_root / 'transforms.json'
        if json_file_path.exists():
            pass
        else:
            with open(str(json_file_path), 'w') as out_file:
                json.dump(out_data, out_file, indent=4)
            print(white_blue('Dumped camera poses (.json) to') + str(json_file_path))

    def render_lighting_envmap(self, render_folder_path):
        '''
        render lighting envmap; separate from render_modality because different camera settings (type and amount)
        '''
        print(blue_text('Rendering lighting_envmap to... by Mitsuba: %s')%str(render_folder_path))
        lighting_global_xyz_list, lighting_global_pts_list = self.os.get_envmap_axes() # each of (env_row, env_col, 3, 3)
        env_row, env_col = self.lighting_params_dict['env_row'], self.lighting_params_dict['env_col']
        env_height, env_width = self.lighting_params_dict['env_height'], self.lighting_params_dict['env_width']
        self.scene.render.resolution_x = env_width
        self.scene.render.resolution_y = env_height
        # [panoramic-cameras] https://docs.blender.org/manual/en/latest/render/cycles/object_settings/cameras.html#panoramic-cameras
        self.cam.data.type = 'PANO'
        # [CyclesCameraSettings]https://docs.blender.org/api/2.80/bpy.types.CyclesCameraSettings.html#bpy.types.CyclesCameraSettings
        self.cam.data.cycles.panorama_type = 'EQUIRECTANGULAR'
        self.cam.data.cycles.latitude_min = 0.

        modal_file_output = self.tree.nodes.new(type="CompositorNodeOutputFile")
        modal_file_output.label = 'Position'
        self.links.new(self.render_layers.outputs['Position'], modal_file_output.inputs[0]) # (self.render_layers.outputs[folder_name], bpy.data.scenes['Scene'].node_tree.nodes["File Output"].inputs[0])
        modal_file_output.base_path = str(render_folder_path)

        T_w_m2b = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], dtype=np.float32) # Mitsuba world to Blender world
        T_c_m2b = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=np.float32)

        bFirstCamera = True
        # bpy.ops.object.select_all(action='DESELECT')
        # # Select the object
        # # https://wiki.blender.org/wiki/Reference/Release_Notes/2.80/Python_API/Scene_and_Object_API
        # bpy.data.objects['Camera'].select_set(True) # Blender 2.8x
        # bpy.ops.object.delete() 
        bpy.context.scene.collection.objects.unlink(self.cam)
        bpy.context.scene.render.use_persistent_data = True
        bpy.context.scene.render.use_multiview = True
        bpy.context.scene.render.views_format = 'MULTIVIEW'
        bpy.context.scene.render.views[0].use = False # 'left
        bpy.context.scene.render.views[1].use = False # 'right

        tic = time.time()
        for frame_id in tqdm(self.os.frame_id_list):
            lighting_global_xyz, lighting_global_pts = lighting_global_xyz_list[frame_id].reshape(-1, 3, 3), lighting_global_pts_list[frame_id].reshape(-1, 3, 3)
            assert lighting_global_xyz.shape[0] == env_row*env_col
            assert lighting_global_pts.shape[0] == env_row*env_col

            im_rendering_path = str(render_folder_path / ('%03d'%(frame_id)))
            self.scene.render.filepath = str(im_rendering_path)

            for env_idx, (xyz, pts) in tqdm(enumerate(zip(lighting_global_xyz, lighting_global_pts))):

                # Create the camera object
                cam_new = bpy.data.objects.new('_%03d'%env_idx, self.cam.data)
                
                pts_b = (T_w_m2b @ (pts.T)).T #  # Mitsuba -> Blender => xyz axes in blender coords
                cam_new.location = pts_b[0].reshape(3, ) # pts_b[0], pts_b[1], pts_b[2] should be the same

                lookatvector_m = xyz[0]; up_m = xyz[2] # follow OpenRooms local hemisphere camera: images/openrooms_hemisphere.jpeg
                R_m = np.stack((np.cross(-up_m, lookatvector_m), -up_m, lookatvector_m), -1)
                assert np.abs(np.linalg.det(R_m)-1) < 1e-5
                R_b = T_w_m2b @ R_m @ T_c_m2b # Mitsuba -> Blender
                euler_ = scipy.spatial.transform.Rotation.from_matrix(R_b).as_euler('xyz')
                cam_new.rotation_euler[0] = euler_[0]
                cam_new.rotation_euler[1] = euler_[1]
                cam_new.rotation_euler[2] = euler_[2]

                bpy.context.scene.collection.objects.link(cam_new)
                bpy.context.scene.camera = cam_new
                # Get the first render view and override it
                # renderView = bpy.context.scene.render.views[0]
                renderView = bpy.context.scene.render.views.new(cam_new.name)

                # Set the camera in the render view
                renderView.name          = cam_new.name
                renderView.camera_suffix = cam_new.name
                renderView.file_suffix   = cam_new.name
                renderView.use = True

                modal_file_output.file_slots[0].path = '%03d_position_'%(frame_id)

            # import ipdb; ipdb.set_trace()
            bpy.ops.render.render(write_still=True)  # render still

        print(blue_text('DONE. (new) %.2f s.'%(time.time()-tic)))