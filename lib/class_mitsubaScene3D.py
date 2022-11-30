from pathlib import Path, PosixPath
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

from tqdm import tqdm
import glob
import scipy
import shutil
from lib.global_vars import mi_variant_dict
import random
random.seed(0)
from lib.utils_io import read_cam_params, normalize_v
import imageio
from PIL import Image
import json
from lib.utils_io import load_matrix, load_img, convert_write_png

import string
# Import the library using the alias "mi"
import mitsuba as mi
# Set the variant of the renderer
# from lib.global_vars import mi_variant
# mi.set_variant(mi_variant)

from lib.utils_misc import blue_text, yellow, get_list_of_keys, white_blue, red
from lib.utils_io import load_matrix, resize_intrinsics
from .class_mitsubaBase import mitsubaBase

from lib.utils_OR.utils_OR_mesh import minimum_bounding_rectangle, mesh_to_contour, load_trimesh, remove_top_down_faces, mesh_to_skeleton, transform_v
from lib.utils_OR.utils_OR_xml import get_XML_root, parse_XML_for_shapes_global
from lib.utils_OR.utils_OR_mesh import loadMesh, computeBox, get_rectangle_mesh
from lib.utils_OR.utils_OR_transform import transform_with_transforms_xml_list
from lib.utils_OR.utils_OR_emitter import load_emitter_dat_world
from lib.utils_OR.utils_OR_lighting import convert_lighting_axis_local_to_global_np, get_ls_np
from lib.utils_misc import get_device

class mitsubaScene3D(mitsubaBase):
    '''
    A class used to visualize/render Mitsuba scene in XML format
    '''
    def __init__(
        self, 
        root_path_dict: dict, 
        scene_params_dict: dict, 
        modality_list: list, 
        im_params_dict: dict={'im_H_load': 480, 'im_W_load': 640, 'im_H_resize': 480, 'im_W_resize': 640, 'spp': 1024}, 
        BRDF_params_dict: dict={}, 
        lighting_params_dict: dict={'env_row': 120, 'env_col': 160, 'SG_num': 12, 'env_height': 16, 'env_width': 32}, # params to load & convert lighting SG & envmap to 
        cam_params_dict: dict={'near': 0.1, 'far': 10.}, 
        shape_params_dict: dict={'if_load_mesh': True}, 
        emitter_params_dict: dict={'N_ambient_rep': '3SG-SkyGrd'},
        mi_params_dict: dict={'if_sample_rays_pts': True, 'if_sample_poses': False}, 
        if_debug_info: bool=False, 
        host: str='', 
    ):

        self.root_path_dict = root_path_dict
        self.PATH_HOME, self.rendering_root, self.xml_scene_root = get_list_of_keys(
            self.root_path_dict, 
            ['PATH_HOME', 'rendering_root', 'xml_scene_root'], 
            [PosixPath, PosixPath, PosixPath]
            )
        self.xml_filename, self.scene_name, self.mitsuba_version, self.intrinsics_path = get_list_of_keys(scene_params_dict, ['xml_filename', 'scene_name', 'mitsuba_version', 'intrinsics_path'], [str, str, str, PosixPath])
        self.split, self.frame_id_list = get_list_of_keys(scene_params_dict, ['split', 'frame_id_list'], [str, list])
        self.mitsuba_version, self.up_axis = get_list_of_keys(scene_params_dict, ['mitsuba_version', 'up_axis'], [str, str])
        self.indexing_based = scene_params_dict.get('indexing_based', 0)
        assert self.mitsuba_version in ['3.0.0', '0.6.0']
        assert self.split in ['train', 'val']
        assert self.up_axis in ['x+', 'y+', 'z+', 'x-', 'y-', 'z-']

        self.scene_path = self.rendering_root / self.scene_name
        self.scene_rendering_path = self.rendering_root / self.scene_name / self.split
        self.scene_rendering_path.mkdir(parents=True, exist_ok=True)
        self.xml_file = self.xml_scene_root / self.scene_name / self.xml_filename

        self.pose_format, pose_file = scene_params_dict['pose_file']
        assert self.pose_format in ['OpenRooms', 'Blender', 'json'], 'Unsupported pose file: '+pose_file
        self.pose_file = self.xml_scene_root / self.scene_name / self.split / pose_file

        self.cam_params_dict = cam_params_dict
        self.shape_params_dict = shape_params_dict
        self.lighting_params_dict = lighting_params_dict
        self.emitter_params_dict = emitter_params_dict
        self.mi_params_dict = mi_params_dict
        self.im_params_dict = im_params_dict

        self.im_H_load, self.im_W_load, self.im_H_resize, self.im_W_resize = get_list_of_keys(im_params_dict, ['im_H_load', 'im_W_load', 'im_H_resize', 'im_W_resize'])
        self.if_resize_im = (self.im_H_load, self.im_W_load) != (self.im_H_resize, self.im_W_resize) # resize modalities (exclusing lighting)
        self.im_target_HW = () if not self.if_resize_im else (self.im_H_resize, self.im_W_resize)
        self.H, self.W = self.im_H_resize, self.im_W_resize

        self.im_sdr_ext = im_params_dict.get('im_sdr_ext', 'png')
        self.im_hdr_ext = im_params_dict.get('im_hdr_ext', 'exr')

        self.near = cam_params_dict.get('near', 0.1)
        self.far = cam_params_dict.get('far', 10.)

        self.host = host
        self.device = get_device(self.host)

        self.modality_list = self.check_and_sort_modalities(list(set(modality_list)))
        self.pcd_color = None
        self.if_loaded_colors = False
        self.if_loaded_shapes = False
        self.if_loaded_layout = False

        ''''
        flags to set
        '''
        self.pts_from = {'mi': False, 'depth': False}
        self.seg_from = {'mi': False, 'seg': False}

        '''
        load everything
        '''
        mitsubaBase.__init__(
            self, 
            device = self.device, 
        )

        self.load_mi_scene(self.mi_params_dict)
        self.load_poses(self.cam_params_dict)

        self.load_modalities_3D()

        self.get_cam_rays(self.cam_params_dict)
        self.process_mi_scene(self.mi_params_dict)

    @property
    def valid_modalities(self):
        return ['layout', 'shapes', 'im_hdr', 'im_sdr']

    def check_and_sort_modalities(self, modalitiy_list):
        modalitiy_list_new = [_ for _ in self.valid_modalities if _ in modalitiy_list]
        for _ in modalitiy_list_new:
            assert _ in self.valid_modalities, 'Invalid modality: %s'%_
        return modalitiy_list_new

    @property
    def if_has_im_sdr(self):
        return hasattr(self, 'im_sdr_list')

    @property
    def if_has_im_hdr(self):
        return hasattr(self, 'im_hdr_list')

    @property
    def if_has_poses(self):
        return hasattr(self, 'pose_list')

    @property
    def if_has_depth_normal(self):
        return all([_ in self.modality_list for _ in ['depth', 'normal']])

    @property
    def if_has_layout(self):
        return all([_ in self.modality_list for _ in ['layout']])

    @property
    def if_has_shapes(self): # objs + emitters
        return all([_ in self.modality_list for _ in ['shapes']])

    @property
    def if_has_mitsuba_scene(self):
        return True

    @property
    def if_has_mitsuba_rays_pts(self):
        return self.mi_params_dict['if_sample_rays_pts']

    @property
    def if_has_mitsuba_segs(self):
        return self.mi_params_dict['if_get_segs']

    @property
    def if_has_mitsuba_all(self):
        return all([self.if_has_mitsuba_scene, self.if_has_mitsuba_rays_pts, self.if_has_mitsuba_segs, ])

    @property
    def if_has_colors(self): # no semantic label colors
        return False

    def load_modalities_3D(self):
        for _ in self.modality_list:
            if _ == 'layout': self.load_layout()
            if _ == 'shapes': self.load_shapes(self.shape_params_dict) # shapes of 1(i.e. furniture) + emitters
            if _ == 'im_sdr': self.load_im_sdr()
            if _ == 'im_hdr': self.load_im_hdr()

    def get_modality(self, modality):
        # if modality in super().valid_modalities:
            # return super(mitsubaScene3D, self).get_modality(modality)

        if 'mi_' in modality:
            assert self.pts_from['mi']

        if modality == 'mi_depth': 
            return self.mi_depth_list
        elif modality == 'mi_normal': 
            return self.mi_normal_global_list
        elif modality in ['mi_seg_area', 'mi_seg_env', 'mi_seg_obj']:
            seg_key = modality.split('_')[-1] 
            return self.mi_seg_dict_of_lists[seg_key]
        else:
            assert False, 'Unsupported modality: ' + modality

    def load_mi_scene(self, mi_params_dict={}):
        '''
        load scene representation into Mitsuba 3
        '''
        variant = mi_params_dict.get('variant', '')
        if variant != '':
            mi.set_variant(variant)
        else:
            mi.set_variant(mi_variant_dict[self.host])

        self.mi_scene = mi.load_file(str(self.xml_file))
        if_also_dump_xml_with_lit_area_lights_only = mi_params_dict.get('if_also_dump_xml_with_lit_area_lights_only', True)
        if if_also_dump_xml_with_lit_area_lights_only:
            from lib.utils_mitsuba import dump_Indoor_area_lights_only_xml_for_mi
            xml_file_lit_up_area_lights_only = dump_Indoor_area_lights_only_xml_for_mi(str(self.xml_file))
            print(blue_text('XML (lit_up_area_lights_only) for Mitsuba dumped to: %s')%str(xml_file_lit_up_area_lights_only))
            self.mi_scene_lit_up_area_lights_only = mi.load_file(str(xml_file_lit_up_area_lights_only))

    def process_mi_scene(self, mi_params_dict={}):
        debug_render_test_image = mi_params_dict.get('debug_render_test_image', False)
        if debug_render_test_image:
            '''
            images/demo_mitsuba_render.png
            '''
            test_rendering_path = self.PATH_HOME / 'mitsuba' / 'tmp_render.exr'
            print(blue_text('Rendering... test frame by Mitsuba: %s')%str(test_rendering_path))
            image = mi.render(self.mi_scene, spp=16)
            mi.util.write_bitmap(str(test_rendering_path), image)
            print(blue_text('DONE.'))

        debug_dump_mesh = mi_params_dict.get('debug_dump_mesh', False)
        if debug_dump_mesh:
            '''
            images/demo_mitsuba_dump_meshes.png
            '''
            mesh_dump_root = self.PATH_HOME / 'mitsuba' / 'meshes_dump'
            if mesh_dump_root.exists(): shutil.rmtree(str(mesh_dump_root))
            mesh_dump_root.mkdir()

            for shape_idx, shape, in enumerate(self.mi_scene.shapes()):
                if not isinstance(shape, mi.llvm_ad_rgb.Mesh): continue
                # print(type(shape), isinstance(shape, mi.llvm_ad_rgb.Mesh))
                shape.write_ply(str(mesh_dump_root / ('%06d.ply'%shape_idx)))

        if_sample_rays_pts = mi_params_dict.get('if_sample_rays_pts', True)
        if if_sample_rays_pts:
            self.mi_sample_rays_pts(self.cam_rays_list)
            self.pts_from['mi'] = True
        
        if_get_segs = mi_params_dict.get('if_get_segs', True)
        if if_get_segs:
            assert if_sample_rays_pts
            self.mi_get_segs(if_also_dump_xml_with_lit_area_lights_only=True)
            self.seg_from['mi'] = True

        if_render_im = self.mi_params_dict.get('if_render_im', False)
        if if_render_im:
            assert False, 'disabled for now; focusing on loading Liwen\'s rendering'
            self.render_im()

    def load_intrinsics(self):
        '''
        -> K: (3, 3)
        '''
        self.K = load_matrix(self.intrinsics_path)
        assert self.K.shape == (3, 3)
        self.im_W_load = int(self.K[0][2] * 2)
        self.im_H_load = int(self.K[1][2] * 2)

        if self.im_W_load != self.W or self.im_H_load != self.H:
            scale_factor = [t / s for t, s in zip((self.H, self.W), (self.im_H_load, self.im_W_load))]
            self.K = resize_intrinsics(self.K, scale_factor)

        if self.pose_format == 'json':
            with open(self.pose_file, 'r') as f:
                self.meta = json.load(f)
            f_xy = 0.5*self.W/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
            assert min(abs(self.K[0][0]-f_xy), abs(self.K[1][1]-f_xy)) < 1e-3, 'computed f_xy is different than read from intrinsics!'

    def load_poses(self, cam_params_dict):
        '''
        pose_list: list of pose matrices (**camera-to-world** transformation), each (3, 4): [R|t] (OpenCV convention: right-down-forward)
        '''
        self.load_intrinsics()
        if hasattr(self, 'pose_list'): return
        if self.mi_params_dict.get('if_sample_poses', False):
            assert False, 'disabled for now; focusing on loading Liwen\'s rendering'
            if_resample = 'n'
            if hasattr(self, 'pose_list'):
                if_resample = input(red("pose_list loaded. Resample pose? [y/n]"))
            if self.pose_file.exists():
                if_resample = input(red("pose file exists: %s. Resample pose? [y/n]"%str(self.pose_file)))
            if if_resample in ['Y', 'y']:
                self.sample_poses(self.mi_params_dict.get('pose_sample_num'), cam_params_dict)
            else:
                print(yellow('ABORTED resample pose.'))
        else:
            if not self.pose_file.exists():
            # if not hasattr(self, 'pose_list'):
                self.get_room_center_pose()

        print(white_blue('[mitsubaScene] load_poses from %s'%str(self.pose_file)))
         
        pose_list = []
        origin_lookatvector_up_list = []

        if self.pose_format == 'OpenRooms':
            '''
            OpenRooms convention (matrices containing origin, lookat, up); The camera coordinates is in OpenCV convention (right-down-forward).
            '''
            cam_params = read_cam_params(self.pose_file)
            assert all([cam_param.shape == (3, 3) for cam_param in cam_params])

            for cam_param in cam_params:
                origin, lookat, up = np.split(cam_param.T, 3, axis=1)
                origin = origin.flatten()
                lookat = lookat.flatten()
                up = up.flatten()
                at_vector = normalize_v(lookat - origin)
                assert np.amax(np.abs(np.dot(at_vector.flatten(), up.flatten()))) < 2e-3 # two vector should be perpendicular

                t = origin.reshape((3, 1)).astype(np.float32)
                R = np.stack((np.cross(-up, at_vector), -up, at_vector), -1).astype(np.float32)
                
                pose_list.append(np.hstack((R, t)))
                origin_lookatvector_up_list.append((origin.reshape((3, 1)), at_vector.reshape((3, 1)), up.reshape((3, 1))))

        elif self.pose_format in ['Blender', 'json']:
            '''
            Blender: 
                Liwen's Blender convention: (N, 2, 3), [t, euler angles]
                Blender x y z == Mitsuba x z -y; Mitsuba x y z == Blender x z -y
            Json:
                Liwen's NeRF poses; processed: in comply with Liwen's IndoorDataset (https://github.com/william122742/inv-nerf/blob/bake/utils/dataset/indoor.py)
            '''
            T_w_b2m = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float32) # Blender world to Mitsuba world; no need if load GT obj (already processed with scale and offset)
            '''
            [NOTE] scene.obj from Liwen is much smaller (s.t. scaling and translation here) compared to scene loaded from scene_v3.xml
            '''
            scale_m2b = np.array([0.206,0.206,0.206], dtype=np.float32).reshape((3, 1))
            trans_m2b = np.array([-0.074684,0.23965,-0.30727], dtype=np.float32).reshape((3, 1))
            t_c2w_b_list, R_c2w_b_list = [], []

            if self.pose_format == 'Blender':
                cam_params = np.load(self.pose_file)
                assert all([cam_param.shape == (2, 3) for cam_param in cam_params])
                T_c_b2m = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]], dtype=np.float32)
                for idx in self.frame_id_list:
                    R_c2w_b_list.append(scipy.spatial.transform.Rotation.from_euler('xyz', [cam_params[idx][1][0], cam_params[idx][1][1], cam_params[idx][1][2]]).as_matrix())
                    t_c2w_b_list.append(cam_param[0].reshape((3, 1)).astype(np.float32))
            
            elif self.pose_format == 'json':
                T_c_b2m = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=np.float32)
                for idx in self.frame_id_list:
                    pose = np.array(self.meta['frames'][idx]['transform_matrix'])[:3, :4].astype(np.float32)
                    R_c2w_b_list.append(np.split(pose, (3,), axis=1)[0])
                    t_c2w_b_list.append(np.split(pose, (3,), axis=1)[1])

            for R_c2w_b, t_c2w_b in zip(R_c2w_b_list, t_c2w_b_list):
                R_c2w_b = R_c2w_b / np.linalg.norm(R_c2w_b, axis=1, keepdims=True)
                assert abs(1.-np.linalg.det(R_c2w_b))<1e-6
                t_c2w_b = (t_c2w_b - trans_m2b) / scale_m2b
                t = T_w_b2m @ t_c2w_b # -> t_c2w_w

                R = T_w_b2m @ R_c2w_b @ T_c_b2m # https://i.imgur.com/nkzfvwt.png

                _, __, at_vector = np.split(R, 3, axis=-1)
                at_vector = normalize_v(at_vector)
                up = normalize_v(-__) # (3, 1)
                assert np.abs(np.sum(at_vector * up)) < 1e-3
                origin = t

                pose_list.append(np.hstack((R, t)))
                origin_lookatvector_up_list.append((origin.reshape((3, 1)), at_vector.reshape((3, 1)), up.reshape((3, 1))))

        self.pose_list = pose_list
        self.origin_lookatvector_up_list = origin_lookatvector_up_list

        print(blue_text('[mistubaScene] DONE. load_poses'))

    def get_cam_rays(self, cam_params_dict={}):
        self.cam_rays_list = self.get_cam_rays_list(self.H, self.W, self.K, self.pose_list)

    def get_room_center_pose(self):
        '''
        generate a single camera, centered at room center and with identity rotation
        '''
        if not self.if_loaded_layout:
            self.load_layout()
        self.pose_list = [np.hstack((
            np.eye(3, dtype=np.float32), ((self.xyz_max+self.xyz_min)/2.).reshape(3, 1)
            ))]

    def sample_poses(self, pose_sample_num: int, cam_params_dict: dict):
        from lib.utils_mitsubaScene_sample_poses import mitsubaScene_sample_poses_one_scene
        assert self.up_axis == 'y+', 'not supporting other axes for now'
        if not self.if_loaded_layout: self.load_layout()

        lverts = self.layout_box_3d_transformed
        boxes = [[bverts, bfaces] for bverts, bfaces, shape in zip(self.bverts_list, self.bfaces_list, self.shape_list_valid) if not shape['is_layout']]
        cads = [[vertices, faces] for vertices, faces, shape in zip(self.vertices_list, self.faces_list, self.shape_list_valid) if not shape['is_layout']]

        cam_params_dict['samplePoint'] = pose_sample_num
        origin_lookat_up_list = mitsubaScene_sample_poses_one_scene(
            scene_dict={
                'lverts': lverts, 
                'boxes': boxes, 
                'cads': cads, 
            }, 
            program_dict={}, 
            param_dict=cam_params_dict, 
            path_dict={},
        ) # [pointLoc; target; up]

        pose_list = []
        origin_lookatvector_up_list = []
        for cam_param in origin_lookat_up_list:
            origin, lookat, up = np.split(cam_param.T, 3, axis=1)
            origin = origin.flatten()
            lookat = lookat.flatten()
            up = up.flatten()
            at_vector = normalize_v(lookat - origin)
            assert np.amax(np.abs(np.dot(at_vector.flatten(), up.flatten()))) < 2e-3 # two vector should be perpendicular
            t = origin.reshape((3, 1)).astype(np.float32)
            R = np.stack((np.cross(-up, at_vector), -up, at_vector), -1).astype(np.float32)
            pose_list.append(np.hstack((R, t)))
            origin_lookatvector_up_list.append((origin.reshape((3, 1)), at_vector.reshape((3, 1)), up.reshape((3, 1))))

        # self.pose_list = pose_list[:pose_sample_num]
        # return

        H, W = self.im_H_load//4, self.im_W_load//4
        scale_factor = [t / s for t, s in zip((H, W), (self.im_H_load, self.im_W_load))]
        K = resize_intrinsics(self.K, scale_factor)
        tmp_cam_rays_list = self.get_cam_rays_list(H, W, K, pose_list)
        normal_costs = []
        depth_costs = []
        normal_list = []
        depth_list = []
        for _, (rays_o, rays_d, ray_d_center) in tqdm(enumerate(tmp_cam_rays_list)):
            rays_o_flatten, rays_d_flatten = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

            xs_mi = mi.Point3f(self.to_d(rays_o_flatten))
            ds_mi = mi.Vector3f(self.to_d(rays_d_flatten))
            rays_mi = mi.Ray3f(xs_mi, ds_mi)
            ret = self.mi_scene.ray_intersect(rays_mi) # https://mitsuba.readthedocs.io/en/stable/src/api_reference.html?highlight=write_ply#mitsuba.Scene.ray_intersect
            rays_v_flatten = ret.t.numpy()[:, np.newaxis] * rays_d_flatten

            mi_depth = np.sum(rays_v_flatten.reshape(H, W, 3) * ray_d_center.reshape(1, 1, 3), axis=-1)
            invalid_depth_mask = np.logical_or(np.isnan(mi_depth), np.isinf(mi_depth))
            mi_depth[invalid_depth_mask] = 0
            depth_list.append(mi_depth)

            mi_normal = ret.n.numpy().reshape(H, W, 3)
            mi_normal[invalid_depth_mask, :] = 0
            normal_list.append(mi_normal)

            mi_normal = mi_normal.astype(np.float32)
            mi_normal_gradx = np.abs(mi_normal[:, 1:] - mi_normal[:, 0:-1])[~invalid_depth_mask[:, 1:]]
            mi_normal_grady = np.abs(mi_normal[1:, :] - mi_normal[0:-1, :])[~invalid_depth_mask[1:, :]]
            ncost = (np.mean(mi_normal_gradx) + np.mean(mi_normal_grady)) / 2.
        
            dcost = np.mean(np.log(mi_depth + 1)[~invalid_depth_mask])

            assert not np.isnan(ncost) and not np.isnan(dcost)
            normal_costs.append(ncost)
            # depth_costs.append(dcost)

        normal_costs = np.array(normal_costs, dtype=np.float32)
        # depth_costs = np.array(depth_costs, dtype=np.float32)
        # normal_costs = (normal_costs - normal_costs.min()) \
        #         / (normal_costs.max() - normal_costs.min())
        # depth_costs = (depth_costs - depth_costs.min()) \
        #         / (depth_costs.max() - depth_costs.min())
        # totalCosts = normal_costs + 0.3 * depth_costs
        totalCosts = normal_costs
        camIndex = np.argsort(totalCosts)[::-1]

        tmp_rendering_path = self.PATH_HOME / 'mitsuba' / 'tmp_sample_poses_rendering'
        if tmp_rendering_path.exists(): shutil.rmtree(str(tmp_rendering_path))
        tmp_rendering_path.mkdir(parents=True, exist_ok=True)
        print(blue_text('Dumping tmp normal and depth by Mitsuba: %s')%str(tmp_rendering_path))
        for i in tqdm(camIndex):
            imageio.imwrite(str(tmp_rendering_path / ('normal_%04d.png'%i)), (np.clip((normal_list[camIndex[i]] + 1.)/2., 0., 1.)*255.).astype(np.uint8))
            imageio.imwrite(str(tmp_rendering_path / ('depth_%04d.png'%i)), (np.clip(depth_list[camIndex[i]] / np.amax(depth_list[camIndex[i]]+1e-6), 0., 1.)*255.).astype(np.uint8))
        print(blue_text('DONE.'))
        # print(normal_costs[camIndex])

        self.pose_list = [pose_list[_] for _ in camIndex[:pose_sample_num]]
        self.origin_lookatvector_up_list = [origin_lookatvector_up_list[_] for _ in camIndex[:pose_sample_num]]

        # if self.pose_file.exists():
        #     txt = input(red("pose_list loaded. Overrite cam.txt? [y/n]"))
        #     if txt in ['N', 'n']:
        #         return
    
        with open(str(self.pose_file), 'w') as camOut:
            cam_poses_write = [origin_lookat_up_list[_] for _ in camIndex[:pose_sample_num]]
            camOut.write('%d\n'%len(cam_poses_write))
            print('Final sampled camera poses: %d'%len(cam_poses_write))
            for camPose in cam_poses_write:
                for n in range(0, 3):
                    camOut.write('%.3f %.3f %.3f\n'%\
                        (camPose[n, 0], camPose[n, 1], camPose[n, 2]))
        print(blue_text('cam.txt written to %s.'%str(self.pose_file)))

    def render_im(self):
        self.spp = self.im_params_dict.get('spp', 1024)
        if_render = 'y'
        im_files = sorted(glob.glob(str(self.scene_rendering_path / 'Image' / '*_*.exr')))
        if len(im_files) > 0:
            if_render = input(red("%d *_*.exr files found at %s. Re-render? [y/n]"))
        if if_render in ['N', 'n']:
            print(yellow('ABORTED rendering by Mitsuba'))
            return
        else:
            shutil.rmtree(str(self.scene_rendering_path / 'Image'))
            self.scene_rendering_path / 'Image'.mkdir(parents=True, exist_ok=True)

        print(blue_text('Rendering RGB to... by Mitsuba: %s')%str(self.scene_rendering_path / 'Image'))
        for i, (origin, lookatvector, up) in tqdm(enumerate(self.origin_lookatvector_up_list)):
            sensor = self.get_sensor(origin, origin+lookatvector, up)
            image = mi.render(self.mi_scene, spp=self.spp, sensor=sensor)
            im_rendering_path = str(self.scene_rendering_path / 'Image' / ('%03d_0001.exr'%i))
            # im_rendering_path = str(self.scene_rendering_path / 'Image' / ('im_%d.rgbe'%i))
            mi.util.write_bitmap(str(im_rendering_path), image)
            '''
            load exr: https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/image_io_and_manipulation.html?highlight=load%20openexr#Reading-an-image-from-disk
            '''

            # im_rgbe = cv2.imread(str(im_rendering_path), -1)
            # dest_path = str(im_rendering_path).replace('.rgbe', '.hdr')
            # cv2.imwrite(dest_path, im_rgbe)
            
            convert_write_png(hdr_image_path='', png_image_path=str(im_rendering_path).replace('.exr', '.png'), if_mask=False, im_hdr=np.array(image))

        print(blue_text('DONE.'))

    def load_im_sdr(self):
        '''
        load im in SDR; RGB, (H, W, 3), [0., 1.]
        '''
        print(white_blue('[mitsubaScene] load_im_sdr'))

        self.im_sdr_file_list = [self.scene_rendering_path / 'Image' / ('%03d_0001.%s'%(i, self.im_sdr_ext)) for i in self.frame_id_list]
        self.im_sdr_list = [load_img(_, expected_shape=(self.im_H_load, self.im_W_load, 3), ext=self.im_sdr_ext, target_HW=self.im_target_HW)/255. for _ in self.im_sdr_file_list]

        print(blue_text('[mitsubaScene] DONE. load_im_sdr'))

    def load_im_hdr(self):
        '''
        load im in HDR; RGB, (H, W, 3), [0., 1.]
        '''
        print(white_blue('[mitsubaScene] load_im_hdr'))

        self.im_hdr_file_list = [self.scene_rendering_path / 'Image' / ('%03d_0001.%s'%(i, self.im_hdr_ext)) for i in self.frame_id_list]
        self.im_hdr_list = [load_img(_, expected_shape=(self.im_H_load, self.im_W_load, 3), ext=self.im_hdr_ext, target_HW=self.im_target_HW) for _ in self.im_hdr_file_list]
        self.hdr_scale_list = [1.] * len(self.im_hdr_list)

        for im_hdr_file, im_hdr in zip(self.im_hdr_file_list, self.im_hdr_list):
            im_sdr_file = Path(str(im_hdr_file).replace(self.im_hdr_ext, self.im_sdr_ext))
            if not im_sdr_file.exists():
                print(yellow('[mitsubaScene] load_im_hdr: converting HDR to SDR and write to disk'), + '-> %s'%str(im_sdr_file))
                convert_write_png(hdr_image_path=str(im_hdr_file), png_image_path=str(im_sdr_file), if_mask=False, scale=1.)

        print(blue_text('[mitsubaScene] DONE. load_im_hdr'))

    def get_sensor(self, origin, target, up):
        from mitsuba import ScalarTransform4f as T
        return mi.load_dict({
            'type': 'perspective',
            'fov': np.arctan(self.K[0][2]/self.K[0][0])/np.pi*180.*2.,
            'fov_axis': 'x',
            'to_world': T.look_at(
                origin=mi.ScalarPoint3f(origin.flatten()),
                target=mi.ScalarPoint3f(target.flatten()),
                up=mi.ScalarPoint3f(up.flatten()),  
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': int(self.spp), 
            },
            'film': {
                'type': 'hdrfilm',
                'width': self.im_W_load,
                'height': self.im_H_load,
                'rfilter': {
                    'type': 'tent',
                },
                'pixel_format': 'rgb',
            },
        })

    def load_shapes(self, shape_params_dict={}):
        '''
        load and visualize shapes (objs/furniture **& emitters**) in 3D & 2D: 
        '''
        if self.if_loaded_shapes: return
        # if_load_obj_mesh = shape_params_dict.get('if_load_obj_mesh', False)
        # if_load_emitter_mesh = shape_params_dict.get('if_load_emitter_mesh', False)
        print(white_blue('[mitsubaScene3D] load_shapes for scene...'))
        root = get_XML_root(self.xml_file)
        shapes = root.findall('shape')
        
        self.shape_list_valid = []
        self.vertices_list = []
        self.faces_list = []
        self.ids_list = []
        self.bverts_list = []
        self.bfaces_list = []

        self.window_list = []
        self.lamp_list = []

        self.xyz_max = np.zeros(3,)-np.inf
        self.xyz_min = np.zeros(3,)+np.inf

        for shape in tqdm(shapes):
            random_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            if_emitter = False; if_window = False; if_area_light = False
            if shape.get('type') != 'obj':
                assert shape.get('type') == 'rectangle'
                '''
                window as rectangle meshes: 
                    images/demo_mitsubaScene_rectangle_windows_1.png
                    images/demo_mitsubaScene_rectangle_windows_2.png
                '''
                transform_m = np.array(shape.findall('transform')[0].findall('matrix')[0].get('value').split(' ')).reshape(4, 4).astype(np.float32) # [[R,t], [0,0,0,1]]
                (vertices, faces) = get_rectangle_mesh(transform_m[:3, :3], transform_m[:3, 3:4])
                _id = 'rectangle_'+random_id
                emitters = shape.findall('emitter')
                if len(emitters) > 0:
                    assert len(emitters) == 1
                    emitter = emitters[0]
                    assert emitter.get('type') == 'area'
                    rgb = emitter.findall('rgb')[0]
                    assert rgb.get('name') == 'radiance'
                    radiance = np.array(rgb.get('value').split(', ')).astype(np.float32).reshape(3,)
                    if_emitter = True; if_area_light = True
                    _id = 'emitter-' + _id
                    emitter_prop = {'intensity': radiance, 'obj_type': 'obj', 'if_lit_up': np.amax(radiance) > 1e-3}
            else:
                if not len(shape.findall('string')) > 0: continue
                _id = shape.findall('ref')[0].get('id')
                # if 'walls' in _id.lower() or 'ceiling' in _id.lower():
                #     continue
                filename = shape.findall('string')[0]; assert filename.get('name') == 'filename'
                obj_path = self.scene_path / filename.get('value') # [TODO] deal with transform
                # if if_load_obj_mesh:
                vertices, faces = loadMesh(obj_path) # based on L430 of adjustObjectPoseCorrectChairs.py
                assert len(shape.findall('emitter')) == 0 # [TODO] deal with object-based emitters

            bverts, bfaces = computeBox(vertices)
            self.vertices_list.append(vertices)
            self.faces_list.append(faces)
            self.bverts_list.append(bverts)
            self.bfaces_list.append(bfaces)
            self.ids_list.append(_id)
            
            shape_dict = {
                'filename': filename.get('value'), 
                'if_in_emitter_dict': if_emitter, 
                'id': _id, 
                'random_id': random_id, 
                # [IMPORTANT] currently relying on definition of walls and ceiling in XML file to identify those, becuase sometimes they can be complex meshes instead of thin rectangles
                'is_wall': 'walls' in _id.lower(), 
                'is_ceiling': 'ceiling' in _id.lower(), 
                'is_layout': 'walls' in _id.lower() or 'ceiling' in _id.lower(), 
            }
            if if_emitter:
                shape_dict.update({'emitter_prop': emitter_prop})
            if if_area_light:
                self.lamp_list.append((shape_dict, vertices, faces))
            self.shape_list_valid.append(shape_dict)

            self.xyz_max = np.maximum(np.amax(vertices, axis=0), self.xyz_max)
            self.xyz_min = np.minimum(np.amin(vertices, axis=0), self.xyz_min)


        self.if_loaded_shapes = True
        print(blue_text('[mitsubaScene3D] DONE. load_shapes: %d total, %d/%d windows lit, %d/%d area lights lit'%(
            len(self.shape_list_valid), 
            len([_ for _ in self.window_list if _[0]['emitter_prop']['if_lit_up']]), len(self.window_list), 
            len([_ for _ in self.lamp_list if _[0]['emitter_prop']['if_lit_up']]), len(self.lamp_list), 
            )))

    def load_layout(self):
        '''
        load and visualize layout in 3D & 2D; assuming room up direction is axis-aligned
        images/demo_layout_mitsubaScene_3D_1.png
        images/demo_layout_mitsubaScene_3D_1_BEV.png # red is layout bbox
        '''

        print(white_blue('[mitsubaScene3D] load_layout for scene...'))
        if self.if_loaded_layout: return
        if not self.if_loaded_shapes: self.load_shapes(self.shape_params_dict)

        vertices_all = np.vstack(self.vertices_list)

        if self.up_axis[0] == 'y':
            self.v_2d = vertices_all[:, [0, 2]]
            # room_height = np.amax(vertices_all[:, 1]) - np.amin(vertices_all[:, 1])
        elif self.up_axis[0] == 'x':
            self.v_2d = vertices_all[:, [1, 3]]
            # room_height = np.amax(vertices_all[:, 0]) - np.amin(vertices_all[:, 0])
        elif self.up_axis[0] == 'z':
            self.v_2d = vertices_all[:, [0, 1]]
            # room_height = np.amax(vertices_all[:, 2]) - np.amin(vertices_all[:, 2])
        # finding minimum 2d bbox (rectangle) from contour
        self.layout_hull_2d = minimum_bounding_rectangle(self.v_2d)
        layout_hull_2d_2x = np.vstack((self.layout_hull_2d, self.layout_hull_2d))
        if self.up_axis[0] == 'y':
            self.layout_box_3d_transformed = np.hstack((layout_hull_2d_2x[:, 0:1], np.vstack((np.zeros((4, 1))+self.xyz_min[1], np.zeros((4, 1))+self.xyz_max[1])), layout_hull_2d_2x[:, 1:2]))
        elif self.up_axis[0] == 'x':
            assert False
        elif self.up_axis[0] == 'z':
            # self.layout_box_3d_transformed = np.hstack((, np.vstack((np.zeros((4, 1)), np.zeros((4, 1))+room_height))))    
            assert False

        print(blue_text('[mitsubaScene3D] DONE. load_layout'))

        self.if_loaded_layout = True

    def load_colors(self):
        '''
        load mapping from obj cat id to RGB
        '''
        self.if_loaded_colors = False
        return