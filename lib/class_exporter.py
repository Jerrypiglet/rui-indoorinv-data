import copy
import pickle
import shutil
from pathlib import Path
import numpy as np
import trimesh
import cv2
from tqdm import tqdm
import mitsuba as mi
import shutil
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et

from lib.utils_OR.utils_OR_mesh import get_rectangle_mesh, get_rectangle_thin_box
from lib.utils_OR.utils_OR_xml import get_XML_root, transformToXml
from lib.utils_io import center_crop
from lib.utils_misc import green, white_red, green_text, yellow, yellow_text, white_blue, blue_text, red, vis_disp_colormap
from lib.utils_OR.utils_OR_xml import xml_rotation_to_matrix_homo
from lib.utils_lieccv22 import x_cam_zq_2_x_cam_rui, x_cam_rui_2_x_cam_zq

from lib.class_openroomsScene2D import openroomsScene2D
from lib.class_openroomsScene3D import openroomsScene3D
from lib.class_mitsubaScene3D import mitsubaScene3D
from lib.class_monosdfScene3D import monosdfScene3D
from lib.class_freeviewpointScene3D import freeviewpointScene3D
from lib.class_matterportScene3D import matterportScene3D
from lib.class_realScene3D import realScene3D
from lib.class_replicaScene3D import replicaScene3D
from lib.class_texirScene3D import texirScene3D
from lib.class_i2sdfScene3D import i2sdfScene3D

class exporter_scene():
    '''
    export scene to following formats:
    - 'monosdf': inv-nerf / MonoSDF
    - 'lieccvee': lieccv22 (Li et al. - 2022 - Physically-Based Editing of Indoor Scene Lighting...)
    '''
    def __init__(
        self, 
        scene_object, 
        format: str='monosdf', 
        modality_list: list=[], 
        if_force: bool=False, 
        scene_shape_file: str='', 
        if_debug_info: bool=False, 
    ):
        
        valid_scene_object_classes = [openroomsScene2D, openroomsScene3D, mitsubaScene3D, monosdfScene3D, freeviewpointScene3D, matterportScene3D, replicaScene3D, realScene3D, texirScene3D, i2sdfScene3D]
        assert type(scene_object) in valid_scene_object_classes, '[%s] has to take an object of %s!'%(self.__class__.__name__, ' ,'.join([str(_.__name__) for _ in valid_scene_object_classes]))

        self.os = scene_object
        self.if_force = if_force
        self.format = format
        assert self.format in ['monosdf', 'lieccv22', 'fvp', 'mitsuba'], '[%s] format: %s is not supported!'%(self.__class__.__name__, self.format)
        self.if_debug_info = if_debug_info
        self.scene_shape_file = scene_shape_file
        
        # self.lieccv22_depth_offset = 0.189
        # self.lieccv22_depth_scale = 0.515

        self.modality_list_export = list(set(modality_list))
        for _ in self.modality_list_export:
            if _ == '': continue
            assert _ in self.valid_modalities, '[%s] modality: %s is not supported!'%(self.__class__.__name__, _)
            
    @property
    def valid_modalities(self):
        return ['im_hdr', 'im_sdr', 'poses', 'im_mask', 'shapes', 'mi_normal', 'mi_depth', 'lighting']
    
    def prepare_check_export(self, scene_export_path: Path):
        if scene_export_path.exists():
            if self.if_force:
                if_reexport = 'Y'
                print(red('scene export path exists:%s . FORCE overwritten.'%str(scene_export_path)))
            else:
                if_reexport = input(red("scene export path exists:%s . Re-export? [y/n]"%str(scene_export_path)))

            if if_reexport in ['y', 'Y']:
                if not self.if_force:
                    if_delete = input(red("Delete? [y/n]"))
                else:
                    if_delete = 'Y'
                if if_delete in ['y', 'Y']:
                    shutil.rmtree(str(scene_export_path), ignore_errors=True)
            else:
                print(red('Aborted export.'))
                return False
        
        # scene_export_path.mkdir(parents=True, exist_ok=True)
        
        return True
        
    def export_monosdf_fvp_mitsuba(self, modality_list={}, appendix='', split='', format='monosdf', if_mask_from_mi: bool=False):
        '''
        export scene to mitsubaScene data structure + monosdf inputs
        and fvp: free viewpoint dataset https://gitlab.inria.fr/sibr/projects/indoor_relighting
        
        - if_mask_from_mi: if True, logical_and loaded im_mask with mask from Mitsuba; if False, use loaded im_mask
        ''' 
        
        assert format in ['monosdf', 'fvp', 'mitsuba'], 'format %s not supported'%format
        scene_name = self.os.scene_name_full if hasattr(self.os, 'scene_name_full') else self.os.scene_name
        scene_export_path = self.os.dataset_root / ('EXPORT_%s'%format) / (scene_name + appendix)
        if split != '':
            scene_export_path = scene_export_path / split
            
        if self.prepare_check_export(scene_export_path) == False:
            return

        modality_list_export = modality_list if len(modality_list) > 0 else self.modality_list_export

        for modality in modality_list_export:
            # assert modality in self.modality_list, 'modality %s not in %s'%(modality, self.modality_list)

            if modality == 'poses':
                if format in ['monosdf', 'mitsuba']: 
                    self.os.export_poses_cam_txt(scene_export_path, cam_params_dict=self.os.CONF.cam_params_dict, frame_num_all=self.os.frame_num_all)
                    '''
                    cameras for MonoSDF
                    '''
                    if format in ['monosdf']: 
                        cameras = {}
                        scale_mat_path = self.os.dataset_root / 'EXPORT_monosdf' / (self.os.scene_name + appendix) / 'scale_mat.npy'
                        scale_mat_path.parent.mkdir(parents=True, exist_ok=True)
                        if split != 'val':
                            scale_mat = np.eye(4).astype(np.float32)
                            if self.os.if_autoscale_scene:
                                print(yellow('Skipped autoscale (following Monosdf) because dataset is already auto scaled.'))
                                (center, scale) = self.os.monosdf_scale_tuple
                            else:
                                poses = [np.vstack((pose, np.array([0., 0., 0., 1.], dtype=np.float32).reshape((1, 4)))) for pose in self.os.pose_list]
                                poses = np.array(poses)
                                assert poses.shape[1:] == (4, 4)
                                min_vertices = poses[:, :3, 3].min(axis=0)
                                max_vertices = poses[:, :3, 3].max(axis=0)
                                center = (min_vertices + max_vertices) / 2.
                                scale = 2. / (np.max(max_vertices - min_vertices) + 3.)
                                
                            print('[pose normalization to unit cube] --center, scale--', center, scale)
                            # we should normalized to unit cube
                            scale_mat[:3, 3] = -center
                            scale_mat[:3 ] *= scale 
                            scale_mat = np.linalg.inv(scale_mat)
                            
                            if_write = True
                            if scale_mat_path.exists():
                                if_overwrite = input(red('scale_mat.npy exists. OVERWRITE? [y/n]'))
                                if if_overwrite in ['y', 'Y'] or self.if_force:
                                    scale_mat_path.unlink()
                                    if_write = True
                                else:
                                    if_write = False
                            if if_write:
                                np.save(str(scale_mat_path), {'scale_mat': scale_mat, 'center': center, 'scale': scale}, allow_pickle=True)
                            else:
                                print(red('SKIPPED DUMP scale_mat.npy'), if_overwrite, self.if_force)
                        else:
                            assert scale_mat_path.exists(), 'scale_mat.npy not found in %s'%str(scale_mat_path)
                            scale_mat_dict = np.load(str(scale_mat_path), allow_pickle=True).item()
                            scale_mat = scale_mat_dict['scale_mat']
                            center = scale_mat_dict['center']
                            scale = scale_mat_dict['scale']

                        for frame_idx, pose in enumerate(self.os.pose_list):
                            if hasattr(self.os, 'K'):
                                K = self.os.K
                            else:
                                assert hasattr(self.os, 'K_list')
                                assert len(self.os.K_list) == len(self.os.pose_list)
                                K = self.os.K_list[frame_idx] # (3, 3)
                                assert K.shape == (3, 3)
                            K = np.hstack((K, np.array([0., 0., 0.], dtype=np.float32).reshape((3, 1))))
                            K = np.vstack((K, np.array([0., 0., 0., 1.], dtype=np.float32).reshape((1, 4))))
                            pose = np.vstack((pose, np.array([0., 0., 0., 1.], dtype=np.float32).reshape((1, 4))))
                            pose = K @ np.linalg.inv(pose)
                            cameras['scale_mat_%d'%frame_idx] = scale_mat
                            cameras['world_mat_%d'%frame_idx] = pose
                            # cameras['split_%d'%frame_idx] = 'train' if _ < self.frame_num-10 else 'val' # 10 frames for val
                            # cameras['frame_id_%d'%frame_idx] = idx if idx < len(mitsuba_scene_dict['train'].pose_list) else idx-len(mitsuba_scene_dict['train'].pose_list)
                        np.savez(str(scene_export_path / 'cameras.npz'), **cameras)
                    
                elif format == 'fvp':
                    '''
                    cameras for fvp; reverse of class_freeviewpointScene3D -> load_poses; dump tp bundle.out
                    '''
                    # bundle.out
                    pose_bundle_file = scene_export_path / 'cameras/bundle.out'
                    pose_bundle_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(str(pose_bundle_file), 'w') as camOut:
                        camOut.write('# Bundle file v0.3\n')
                        camOut.write('%d %d\n'%(self.os.frame_num, 0)) # <num_cameras> <num_points>   [two integers]
                        for frame_idx, frame_id in tqdm(enumerate(self.os.frame_id_list)):
                            K = self.os._K(frame_idx)
                            f = K[0][0]
                            # assert K[0][0] == K[1][1], 'focal length of x and y not equal: %f, %f'%(K[0][0], K[1][1])
                            assert K[0][1] == 0. and K[1][0] == 0. # no redial distortion
                            camOut.write('%.6f %d %d\n'%(f, 0, 0)) # <f> <k1> <k2>   [the focal length, followed by two radial distortion coeffs]
                            
                            # [!!!] assert if_scale_scene = 1.0
                            R, t = self.os.pose_list[frame_idx][:3, :3].copy(), self.os.pose_list[frame_idx][:3, 3].copy()
                            # if self.extra_transform is not None:
                            #     R = R @ (self.extra_transform.T)
                            #     t = self.extra_transform @ t
                            if hasattr(self.os, 'scene_scale'):
                                t = t * self.os.scene_scale
                            R = R @ np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=np.float32) # OpenCV -> OpenGL
                            R_ = R.T
                            t_ = -R_ @ t
                            assert np.isclose(np.linalg.det(R_), 1.)
                            for R_row in R_: # <R> [a 3x3 matrix representing the camera rotation]
                                camOut.write('%.6f %.6f %.6f\n'%(R_row[0], R_row[1], R_row[2]))
                            # <t> [a 3-vector describing the camera translation]
                            camOut.write('%.6f %.6f %.6f\n'%(t_.flatten()[0], t_.flatten()[1], t_.flatten()[2]))

                            lookat_file_path = scene_export_path / 'cameras' / ('%d_cameras.lookat'%frame_idx)
                            origin = t.flatten()
                            lookatvector = (R @ np.array([[0.], [0.], [-1.]], dtype=np.float32)).flatten()
                            up = (R @ np.array([[0.], [1.], [0.]], dtype=np.float32)).flatten()
                            
                            lookat_str = '%05d.exr'%frame_idx
                            lookat_str += (' -D origin=%.4f,%.4f,%.4f'%(origin[0], origin[1], origin[2]))
                            lookat_str += (' -D target=%.4f,%.4f,%.4f'%((origin+lookatvector)[0], (origin+lookatvector)[1], (origin+lookatvector)[2]))
                            lookat_str += (' -D up=%.4f,%.4f,%.4f'%(up[0], up[1], up[2]))
                            fy = self.os._K(frame_idx)[1][1]
                            fov_y = np.arctan(0.5 * self.os.H / fy) * 2. / np.pi * 180.
                            lookat_str += ' -D fovy=%.2f -D clip=0.001000,1000.000000'%fov_y
                            with open(str(lookat_file_path), 'a') as lookat_file:
                                lookat_file.write(lookat_str + '\n')
                else:
                    raise NotImplementedError

            if modality == 'im_hdr':
                assert self.os.if_has_im_hdr
                file_str = {'monosdf': 'Image/%03d_0001.exr', 'mitsuba': 'Image/%03d_0001.exr', 'fvp': 'images/%05d.exr'}[format]
                (scene_export_path / file_str).parent.mkdir(parents=True, exist_ok=True)
                for frame_idx, frame_id in enumerate(self.os.frame_id_list):
                    im_hdr_export_path = scene_export_path / (file_str%frame_idx)
                    hdr_scale = self.os.hdr_scale_list[frame_idx]
                    cv2.imwrite(str(im_hdr_export_path), hdr_scale * self.os.im_hdr_list[frame_idx][:, :, [2, 1, 0]])
                    print(blue_text('HDR image %d exported to: %s'%(frame_id, str(im_hdr_export_path))))
            
            if modality == 'im_sdr':
                assert self.os.if_has_im_sdr
                file_str = {'monosdf': 'Image/%03d_0001.png', 'mitsuba': 'Image/%03d_0001.png', 'fvp': 'images/%05d.jpg'}[format]
                (scene_export_path / file_str).parent.mkdir(parents=True, exist_ok=True)
                for frame_idx, frame_id in enumerate(self.os.frame_id_list):
                    im_sdr_export_path = scene_export_path / (file_str%frame_idx)
                    cv2.imwrite(str(im_sdr_export_path), (np.clip(self.os.im_sdr_list[frame_idx][:, :, [2, 1, 0]], 0., 1.)*255.).astype(np.uint8))
                    print(blue_text('SDR image %d exported to: %s'%(frame_id, str(im_sdr_export_path))))
                    # print(self.os.modality_file_list_dict['im_sdr'][frame_idx])
            
            if modality == 'albedo': # HDR
                assert hasattr(self.os, 'albedo_list')
                file_str = 'DiffCol/%03d.exr'
                (scene_export_path / file_str).parent.mkdir(parents=True, exist_ok=True)
                for frame_idx, frame_id in enumerate(self.os.frame_id_list):
                    albedo_export_path = scene_export_path / (file_str%frame_idx)
                    cv2.imwrite(str(albedo_export_path), (np.clip(self.os.albedo_list[frame_idx][:, :, [2, 1, 0]], 0., 1.)).astype(np.float32))
                    print(blue_text('Albedo HDR image %d exported to: %s'%(frame_id, str(albedo_export_path))))

            if modality == 'roughness': # [0., 1.]
                assert hasattr(self.os, 'roughness_list')
                file_str = 'Roughness/%03d.exr'
                (scene_export_path / file_str).parent.mkdir(parents=True, exist_ok=True)
                for frame_idx, frame_id in enumerate(self.os.frame_id_list):
                    roughness_export_path = scene_export_path / (file_str%frame_idx)
                    cv2.imwrite(str(roughness_export_path), (np.clip(self.os.roughness_list[frame_idx], 0., 1.)).astype(np.float32))
                    print(blue_text('Roughness image %d exported to: %s'%(frame_id, str(roughness_export_path))))
            
            if 'mi_seg' in modality:
                _mod = modality.split('_')[-1]
                assert _mod in ['env', 'obj', 'area']
                assert self.os.seg_from['mi']
                file_str = 'MiSeg' + {'env': 'Env', 'obj': 'Obj', 'area': 'Area'}[_mod] + '/%03d.png'
                (scene_export_path / file_str).parent.mkdir(parents=True, exist_ok=True)
                for frame_idx, frame_id in enumerate(self.os.frame_id_list):
                    mi_seg_mod_export_path = scene_export_path / (file_str%frame_idx)
                    cv2.imwrite(str(mi_seg_mod_export_path), (self.os.mi_seg_dict_of_lists[_mod][frame_idx]*255).astype(np.uint8))
                    print(blue_text('mi_seg_%s image %d exported to: %s'%(_mod, frame_id, str(mi_seg_mod_export_path))))

            if modality == 'mi_normal':
                '''
                -> normal: npy [0, 1], [3, H, W], CAMERA coords: OpenCV
                '''
                assert format in ['monosdf', 'mitsuba']
                (scene_export_path / 'MiNormal').mkdir(parents=True, exist_ok=True)
                (scene_export_path / '_MiNormalOpenCV_OVERLAY').mkdir(parents=True, exist_ok=True)
                assert self.os.pts_from['mi']
                for frame_idx, frame_id in enumerate(self.os.frame_id_list):
                    _mi_normal_export_path = scene_export_path / 'MiNormal' / ('%03d_0001.png'%frame_idx)
                    _mi_normal = self.os.mi_normal_opencv_list[frame_idx]/2.+0.5 # [-1, 1] -> [0, 1]
                    assert _mi_normal.shape == (self.os._H(frame_idx), self.os._W(frame_idx), 3)
                    cv2.imwrite(str(_mi_normal_export_path), (np.clip(_mi_normal[:, :, [2, 1, 0]], 0., 1.)*255.).astype(np.uint8))
                    print(blue_text('Mitsuba mi_normal (vis) (OpenCV) %d exported to: %s'%(frame_id, str(_mi_normal_export_path))))
                    
                    mi_normal_export_path = scene_export_path / 'MiNormal' / ('%03d_0001.npy'%frame_idx)
                    mi_normal = np.clip(_mi_normal.transpose(2, 0, 1), 0., 1.)
                    assert mi_normal.shape == (3, self.os._H(frame_idx), self.os._W(frame_idx))
                    np.save(str(mi_normal_export_path), mi_normal)
                    print(blue_text('Mitsuba mi_normal %d exported to: %s'%(frame_id, str(mi_normal_export_path))))
                    
                    _mi_normal_overlay = self.os.im_sdr_list[frame_idx].copy()
                    _mi_normal = self.os.mi_normal_opencv_list[frame_idx]/2.+0.5
                    _mi_normal_overlay = _mi_normal_overlay * 0.5 + _mi_normal * 0.5
                    assert _mi_normal_overlay.shape == (self.os._H(frame_idx), self.os._W(frame_idx), 3)
                    _mi_normal_overlay_export_path = scene_export_path / '_MiNormalOpenCV_OVERLAY' / ('%03d_0001.png'%frame_idx)
                    cv2.imwrite(str(_mi_normal_overlay_export_path), (np.clip(_mi_normal_overlay[:, :, [2, 1, 0]], 0., 1.)*255.).astype(np.uint8))
            
            if modality == 'mi_depth':
                '''
                In un-normalized space!!
                Multiply by scale_mat_dict['scale'] -> monosdf depth
                '''
                assert format in ['monosdf', 'mitsuba']
                (scene_export_path / 'MiDepth').mkdir(parents=True, exist_ok=True)
                assert self.os.pts_from['mi']
                for frame_idx, frame_id in enumerate(self.os.frame_id_list):
                    mi_depth_vis_export_path = scene_export_path / 'MiDepth' / ('%03d_0001.png'%frame_idx)
                    mi_depth = self.os.mi_depth_list[frame_idx].squeeze()
                    if np.sum(~self.os.mi_invalid_depth_mask_list[frame_idx]) == 0:
                        depth_normalized = mi_depth
                    else:
                        depth_normalized, depth_min_and_scale = vis_disp_colormap(mi_depth, normalize=True, valid_mask=~self.os.mi_invalid_depth_mask_list[frame_idx])
                    cv2.imwrite(str(mi_depth_vis_export_path), depth_normalized)
                    print(blue_text('depth (vis) %d exported to: %s'%(frame_id, str(mi_depth_vis_export_path))))
                    mi_depth_npy_export_path = scene_export_path / 'MiDepth' / ('%03d_0001.npy'%frame_idx)
                    np.save(str(mi_depth_npy_export_path), mi_depth)
                    print(blue_text('depth (npy) %d exported to: %s'%(frame_id, str(mi_depth_npy_export_path))))

            if modality == 'normal':
                '''
                -> normal: npy [0, 1], [3, H, W], CAMERA coords: OpenCV
                '''
                assert hasattr(self.os, 'normal_list')
                (scene_export_path / 'normal').mkdir(parents=True, exist_ok=True)
                (scene_export_path / '_normal_OVERLAY').mkdir(parents=True, exist_ok=True)
                for frame_idx, frame_id in enumerate(self.os.frame_id_list):
                    normal_export_path = scene_export_path / 'normal' / ('%03d_0001.npy'%frame_idx)
                    normal = (self.os.normal_list[frame_idx].transpose(2, 0, 1))/2.+0.5
                    assert normal.shape == (3, self.os._H(frame_idx), self.os._W(frame_idx))
                    np.save(str(normal_export_path), normal)
                    print(blue_text('Normal %d exported to: %s'%(frame_id, str(normal_export_path))))
                    
                    _normal = self.os.normal_list[frame_idx]/2.+0.5
                    assert _normal.shape == (self.os._H(frame_idx), self.os._W(frame_idx), 3)
                    normal_vis_export_path = scene_export_path / 'normal' / ('%03d_0001.png'%frame_idx)
                    cv2.imwrite(str(normal_vis_export_path), (np.clip(_normal[:, :, [2, 1, 0]], 0., 1.)*255.).astype(np.uint8))
                    
                    _normal_overlay = self.os.im_sdr_list[frame_idx].copy()
                    _normal_overlay = _normal_overlay * 0.5 + _normal * 0.5
                    assert _normal_overlay.shape == (self.os._H(frame_idx), self.os._W(frame_idx), 3)
                    _normal_overlay_export_path = scene_export_path / '_normal_OVERLAY' / ('%03d_0001.png'%frame_idx)
                    cv2.imwrite(str(_normal_overlay_export_path), (np.clip(_normal_overlay[:, :, [2, 1, 0]], 0., 1.)*255.).astype(np.uint8))
            
            if modality == 'depth':
                '''
                In un-normalized space!!
                Multiply by scale_mat_dict['scale'] -> monosdf depth
                '''
                assert hasattr(self.os, 'depth_list')
                (scene_export_path / 'depth').mkdir(parents=True, exist_ok=True)
                for frame_idx, frame_id in enumerate(self.os.frame_id_list):
                    depth_vis_export_path = scene_export_path / 'Depth' / ('%03d_0001.png'%frame_idx)
                    depth = self.os.depth_list[frame_idx].squeeze()
                    if hasattr(self.os, 'im_mask_list'):
                        valid_mask = self.os.im_mask_list[frame_idx].squeeze()
                    else:
                        valid_mask = None
                    depth_normalized, depth_min_and_scale = vis_disp_colormap(depth, normalize=True, valid_mask=valid_mask)
                    cv2.imwrite(str(depth_vis_export_path), depth_normalized)
                    print(blue_text('depth (vis) %d exported to: %s'%(frame_id, str(depth_vis_export_path))))
                    
                    depth_npy_export_path = scene_export_path / 'depth' / ('%03d_0001.npy'%frame_idx)
                    np.save(str(depth_npy_export_path), depth)
                    print(blue_text('depth (npy) %d exported to: %s'%(frame_id, str(depth_npy_export_path))))

            if modality == 'im_mask':
                file_str = {'monosdf': 'ImMask/%03d_0001.png', 'mitsuba': 'ImMask/%03d_0001.png', 'fvp': 'images/%08d_mask.png'}[format]
                (scene_export_path / file_str).parent.mkdir(parents=True, exist_ok=True)
                
                if_undist_mask = False
                if if_mask_from_mi:
                    if not self.os.pts_from['mi']:
                        print(yellow('Skipped exporting im_mask because mi_depth is not available.'))
                        continue
                    if hasattr(self.os, 'im_mask_list'):
                        assert len(self.os.im_mask_list) == len(self.os.mi_invalid_depth_mask_list)
                    if hasattr(self.os, 'if_undist'):
                        if self.os.if_undist:
                            assert hasattr(self.os, 'im_undist_mask_list')
                            assert len(self.os.im_undist_mask_list) == len(self.os.mi_invalid_depth_mask_list)
                            if_undist_mask = True
                        
                for frame_idx, frame_id in enumerate(self.os.frame_id_list):
                    im_mask_export_path = scene_export_path / (file_str%frame_idx)
                    mask_source_list = []
                    if if_mask_from_mi:
                        mi_invalid_depth_mask = self.os.mi_invalid_depth_mask_list[frame_idx]
                        assert mi_invalid_depth_mask.dtype == bool
                        im_mask_export = ~mi_invalid_depth_mask
                        mask_source_list += ['mi']
                    else:
                        mask_source_list += []
                        im_mask_export = np.ones((self.os._H(frame_idx), self.os._W(frame_idx)), dtype=bool)
                        assert hasattr(self.os, 'im_mask_list')
                        
                    if hasattr(self.os, 'im_mask_list'):
                        im_mask_ = self.os.im_mask_list[frame_idx]
                        assert im_mask_.dtype == bool
                        assert im_mask_export.shape == im_mask_.shape, 'invalid depth mask shape %s not equal to im_mask shape %s'%(mi_invalid_depth_mask.shape, im_mask_.shape)
                        im_mask_export = np.logical_and(im_mask_export, im_mask_)
                        mask_source_list.append('im_mask')
                        
                    if if_undist_mask:
                        im_mask_ = self.os.im_undist_mask_list[frame_idx]
                        assert im_mask_.dtype == bool
                        assert im_mask_export.shape == im_mask_.shape, 'invalid depth mask shape %s not equal to im_undist_mask shape %s'%(mi_invalid_depth_mask.shape, im_mask_.shape)
                        im_mask_export = np.logical_and(im_mask_export, im_mask_)
                        mask_source_list.append('im_undist_mask')
                        
                    print('Exporting im_mask from %s'%(' && '.join(mask_source_list)))
                    cv2.imwrite(str(im_mask_export_path), (im_mask_export*255).astype(np.uint8))
                    print(blue_text('Mask image (for valid depths) %s exported to: %s'%(frame_id, str(im_mask_export_path))))
                    
            if modality == 'matseg':
                assert hasattr(self.os, 'matseg_list')
                file_str = {'monosdf': 'MatSeg/%03d.npy', 'mitsuba': 'MatSeg/%03d.npy', 'fvp': 'MatSeg/%03d.npy'}[format]
                (scene_export_path / file_str).parent.mkdir(parents=True, exist_ok=True)
                for frame_idx, frame_id in enumerate(self.os.frame_id_list):
                    matseg_export_path = scene_export_path / (file_str%frame_idx)
                    matseg = self.os.matseg_list[frame_idx]['mat_aggre_map'] # (H, W), int32; [0, 1, ..., num_mat_masks], 0 for invalid region
                    assert np.amax(matseg) < 255 # otherwise uint8 does not work
                    np.save(str(matseg_export_path), matseg.astype(np.uint8))
                    print(blue_text('MatSeg %d exported to: %s'%(frame_id, str(matseg_export_path))))
            
            if modality == 'lighting': 
                outLight_file_list = [_ for _ in self.os.scene_path.iterdir() if _.stem.startswith('outLight')]
                if len(outLight_file_list) > 0:
                    print(white_blue('Found %d outLight files at'%len(outLight_file_list)), str(self.os.scene_path))
                    (scene_export_path / 'lightings').mkdir(parents=True, exist_ok=True)
                    for outLight_file in outLight_file_list:
                        root = get_XML_root(str(outLight_file))
                        root = copy.deepcopy(root)
                        shapes = root.findall('shape')
                        assert len(shapes) > 0, 'No shapes found in %s; double check you XML file (e.g. did you miss headings)?'%str(outLight_file)
                        
                        for shape in tqdm(shapes):
                            emitters = shape.findall('emitter')
                            assert len(emitters) == 1
                            # if shape.get('type') != 'obj':
                            assert shape.get('type') in ['obj', 'rectangle', 'sphere']
                            #     assert shape.get('type') == 'rectangle'
                                
                            transform_item = shape.findall('transform')[0]
                            transform_accu = np.eye(4, dtype=np.float32)
                            
                            if hasattr(self.os, 'if_reorient_shape') and self.os.if_reorient_shape:
                                assert hasattr(self.os, 'reorient_transform')
                                transform_accu = self.os.reorient_transform @ transform_accu

                            if len(transform_item.findall('rotate')) > 0:
                                rotate_item = transform_item.findall('rotate')[0]
                                _r_h = xml_rotation_to_matrix_homo(rotate_item)
                                transform_accu = _r_h @ transform_accu
                                
                            if len(transform_item.findall('matrix')) > 0:
                                _transform = [_ for _ in transform_item.findall('matrix')[0].get('value').split(' ') if _ != '']
                                transform_matrix = np.array(_transform).reshape(4, 4).astype(np.float32)
                                transform_accu = transform_matrix @ transform_accu # [[R,t], [0,0,0,1]]

                            if self.os.extra_transform is not None:
                                transform_accu = self.os.extra_transform_homo @ transform_accu
                                
                            if shape.get('type') == 'rectangle':
                                (_vertices, _light_faces) = get_rectangle_mesh(transform_accu[:3, :3], transform_accu[:3, 3:4])
                                light_trimesh = trimesh.Trimesh(vertices=_vertices, faces=_light_faces-1)
                            elif shape.get('type') == 'obj':
                                light_mesh_path = self.os.scene_path / shape.findall('string')[0].get('value')
                                assert light_mesh_path.exists()
                                light_mesh = trimesh.load_mesh(str(light_mesh_path))
                                _vertices, _light_faces = light_mesh.vertices, light_mesh.faces
                                _vertices = (transform_accu[:3, :3] @ _vertices.T + transform_accu[:3, 3:4]).T
                                light_trimesh = trimesh.Trimesh(vertices=_vertices, faces=_light_faces)
                            elif shape.get('type') == 'sphere':
                                light_mesh = trimesh.creation.icosphere(subdivisions=3)
                                _vertices, _light_faces = light_mesh.vertices, light_mesh.faces
                                _vertices = (transform_accu[:3, :3] @ _vertices.T + transform_accu[:3, 3:4]).T
                                light_trimesh = trimesh.Trimesh(vertices=_vertices, faces=_light_faces)
                                
                            light_mesh_path = scene_export_path / 'lightings' / (outLight_file.stem + '_mesh.obj')
                            light_trimesh.export(str(light_mesh_path))
                            
                            # [TODO] dump mesh obj file as well so that fvp can access it. And change to abs path in outLight XML.
                            
                            if self.os.extra_transform is not None:
                                transform_matrix = self.os.extra_transform_homo @ transform_matrix
                            _transform_matrix_new = ' '.join(['%f'%_ for _ in transform_matrix.reshape(-1)])
                            shape.findall('transform')[0].findall('matrix')[0].set('value', _transform_matrix_new)
                            
                            # write another *_scale.txt, and set emitter max radiance to 1.
                            assert len(shape.findall('emitter')) == 1
                            assert shape.findall('emitter')[0].get('type') == 'area'
                            _rad_item = shape.findall('emitter')[0].findall('rgb')[0]
                            assert _rad_item.get('name') == 'radiance'
                            _rad = [float(_) for _ in _rad_item.get('value').split(',')]
                            assert len(_rad) == 3
                            _rad_max = max(_rad)
                            # _rad_max = 1.
                            _rad_item.set('value', ' '.join(['%.2f'%(_/(_rad_max+1e-6)) for _ in _rad]))
                            
                            xmlString = transformToXml(root)
                            xml_filepath = scene_export_path / 'lightings' / outLight_file.name
                            (scene_export_path / xml_filepath).parent.mkdir(parents=True, exist_ok=True)
                            with open(str(xml_filepath), 'w') as xmlOut:
                                xmlOut.write(xmlString)
                            
                            txt_filepath = scene_export_path / 'lightings' / (outLight_file.stem + '_scale.txt')
                            with open(str(txt_filepath), 'w') as txtOut:
                                txtOut.write('%.2f'%(_rad_max))
                            
                            print(blue_text('lighting exported to: %s'%str(xml_filepath)))
                            # else:
                            #     assert False, 'todo: deal with obj emitter'
                                # if not len(shape.findall('string')) > 0: continue
                                # filename = shape.findall('string')[0]; assert filename.get('name') == 'filename'
                                # obj_path = self.scene_path / filename.get('value') # [TODO] deal with transform
                                # shape_trimesh = trimesh.load_mesh(str(obj_path), process=False, maintain_order=True)
                                # vertices, faces = np.array(shape_trimesh.vertices), np.array(shape_trimesh.faces)+1

            if modality == 'shapes': 
                if not self.os.if_loaded_shapes:
                    print(yellow('Skipping shapes export since shapes not loaded.'))
                    continue
                shape_list = []
                file_str = {'monosdf': 'scene%s.obj'%appendix, 'mitsuba': 'scene%s.obj'%appendix, 'fvp': 'meshes/recon.ply'}[format]
                (scene_export_path / file_str).parent.mkdir(parents=True, exist_ok=True)
                shape_export_path = scene_export_path / file_str
                
                for vertices, faces in zip(self.os.vertices_list, self.os.faces_list):
                    shape_list.append(trimesh.Trimesh(vertices, faces-1, process=True, maintain_order=True))
                shape_tri_mesh = trimesh.util.concatenate(shape_list)
                
                # trimesh.repair.fill_holes(shape_tri_mesh)
                # trimesh.repair.fix_winding(shape_tri_mesh)
                # trimesh.repair.fix_inversion(shape_tri_mesh)
                # trimesh.repair.fix_normals(shape_tri_mesh)
                
                shape_tri_mesh.export(str(shape_export_path))
                print('-->', shape_tri_mesh.faces[19531])
                print(white_blue('Shape exported to: %s'%str(shape_export_path)))
                if not shape_tri_mesh.is_watertight:
                    # slightly expand the shape to get bigger convex hull to avoid overlap faces -> flickering in Meshlab
                    _v = shape_tri_mesh.vertices
                    _v_mean = np.mean(_v, axis=0, keepdims=True)
                    shape_tri_mesh_tmp = trimesh.Trimesh(vertices=(_v - _v_mean) * 1.1 + _v_mean, faces=shape_tri_mesh.faces)
                    shape_tri_mesh_convex = trimesh.convex.convex_hull(shape_tri_mesh_tmp)
                    
                    shape_tri_mesh_convex.export(str(shape_export_path.parent / ('%s_hull%s.obj'%(shape_export_path.stem, appendix))))
                    shape_tri_mesh_fixed = trimesh.util.concatenate([shape_tri_mesh, shape_tri_mesh_convex])
                    if_fixed_water_tight = False
                    
                    if format in ['monosdf']:
                        shape_tri_mesh.export(str(shape_export_path))
                        
                        # if_fixed_water_tight = True
                        # print(red('Overwriting with mesh + hull: %s'%str(shape_export_path)))
                        # shape_tri_mesh_fixed.export(str(shape_export_path))

                    elif format == 'fvp': 
                        # scale.txt
                        scene_scale = self.os.scene_scale if hasattr(self.os, 'scene_scale') else 1.
                        with open(str(scene_export_path / 'scale.txt'), 'w') as camOut:
                            camOut.write('%.4f'%scene_scale)
                        # overwrite the original mesh
                        print(red('Overwriting with mesh + hull: %s'%str(shape_export_path)))
                        shape_tri_mesh_fixed.export(str(shape_export_path))

                    else:
                        pass

                    if if_fixed_water_tight:
                        print(yellow('Mesh is not watertight. Filled holes and added convex hull: -> %s%s.obj, %s_hull%s.obj, %s_fixed%s.obj'%(shape_export_path.name, appendix, shape_export_path.name, appendix, shape_export_path.name, appendix)))
                
                '''
                images/demo_fvp_scale.png
                - measurement: 1.39, real size in meters: probably 0.6
                -> scale = 1.39 / 0.6 = 2.3166666666666664
                '''
                if self.os.if_autoscale_scene and format == 'fvp':
                    print(red('DONT GOTGETR TO SET CORRECT SCLAE IN SCALE.TXT (equal to 1m object in the scene scale)'))
                    print('Check the demo as described here: https://gitlab.inria.fr/sibr/projects/indoor_relighting (search scale.txt)')
                    input(red('Type Y to acknowledge'))

    def export_lieccv22(
        self, 
        modality_list=[], 
        appendix='', 
        dataset_name='', 
        split='', 
        center_crop_HW=None, 
        assert_shape=None, 
        window_area_emitter_id_list: list=[], 
        merge_lamp_id_list: list=[], 
        emitter_thres: float=0., 
        if_no_gt_appendix: bool=False, 
        BRDF_results_folder: str=None,
        ):
        '''
        export lieccv22 scene to Zhengqin's ECCV'22 format
        
        - window_area_emitter_id: id for the are light which is acrually a window...
        - merge_lamp_id_list: list of lamp ids to be merged into one
        - emitter_thres: use this to get emitter mask, if no ground truth emitter is available
        
        '''
        assert center_crop_HW is None, 'rendered to target sizes for now: no extra center crop'
        scene_export_path = self.os.dataset_root / self.os.scene_name / 'EXPORT_lieccv22' / split
        if self.prepare_check_export(scene_export_path) == False:
            return
        

        
        modality_list_export = modality_list if len(modality_list) > 0 else self.modality_list_export
        
        if assert_shape is not None:
            assert assert_shape == (self.os.H, self.os.W), 'assert_shape %s not equal to (H, W) %s'%(assert_shape, (self.os.H, self.os.W))
            
        lamp_dict = {_['id'].replace('emitter-', ''): _ for _ in self.os.lamp_list if _['emitter_prop']['if_lit_up']} # only lit up lamps
        window_dict = {_['id'].replace('emitter-', ''): _ for _ in self.os.window_list}
        if window_area_emitter_id_list != []:
            for window_area_emitter_id in window_area_emitter_id_list:
                assert window_area_emitter_id in lamp_dict, 'window_area_emitter_id %s not in lamp_dict (keys: %s); did you assign IDs to emitters in XML?'%(window_area_emitter_id, '-'.join(lamp_dict.keys()))
                window_dict.update({window_area_emitter_id: lamp_dict[window_area_emitter_id]})
                lamp_dict.pop(window_area_emitter_id)
        if merge_lamp_id_list != []:
            assert [_ in lamp_dict for _ in merge_lamp_id_list]
            new_lamp = [lamp_dict[_] for _ in merge_lamp_id_list]
            enw_lamp_id = '+'.join(merge_lamp_id_list)
            lamp_dict.update({enw_lamp_id: new_lamp})
            for lamp_id in merge_lamp_id_list:
                lamp_dict.pop(lamp_id)
        if window_area_emitter_id_list != [] or merge_lamp_id_list != []:
            print(yellow('🛋️ Found %d lamps and %d windows in GROUND TRUTH'%(len(lamp_dict), len(window_dict))))
        else:
            print(yellow('💡Using emitter_thres %.2f to get emitter mask, if no ground truth emitter is available'%(emitter_thres)))
        
        frame_export_path_list = []
        for frame_idx, frame_id in enumerate(self.os.frame_id_list):
            frame_export_path = self.os.dataset_root / 'EXPORT_lieccv22' / split / (self.os.scene_name + '_frame%d'%frame_id + appendix) / 'input'
            frame_export_path.mkdir(parents=True, exist_ok=True)
            frame_export_path_list.append(frame_export_path)
            
            '''
            in case depth is normalized in ZQ's processing; in this case we need to transform the inseted lights
            '''
            depth_scale_mul = None
            depth_scale_path = Path(frame_export_path) / 'depth_scale_mul.npy'
            if depth_scale_path.exists():
                print(yellow('Depth normalization file exists; applying normalization to [light] goemetry and location.'), str(frame_export_path))
                depth_scale_mul = np.load(depth_scale_path)
            
            '''
            try not change ZQ's fov because his models were trained with his fixed fov, and his renderers only supports his fov (i.e. equal fx fy, and centered optical center)
            '''
            # fov_x = self.os.meta['camera_angle_x'] / np.pi * 180.
            # fov_x = self.os.meta['camera_angle_x'] / np.pi * 180.
            # fx = self.os._K(frame_idx)[0][0]
            # fy = self.os._K(frame_idx)[1][1]
            # fov_x = np.arctan(0.5 * self.os.W / fx) * 2. / np.pi * 180.
            # fov_y = np.arctan(0.5 * self.os.H / fy) * 2. / np.pi * 180.
            
            # np.save(frame_export_path / 'fov_xy.npy', {'fov_x': fov_x, 'fov_y': fov_y}) # full fov, in angles
            
            IF_CREATED_EDIT_FOLDER = False
            
            for modality in modality_list_export:
                '''
                export png images
                '''
                if modality == 'im_sdr':
                    im_sdr_export_path = frame_export_path / 'im.png'
                    im_sdr = (np.clip(self.os.im_sdr_list[frame_idx][:, :, [2, 1, 0]], 0., 1.)*255.).astype(np.uint8)
                    im_sdr = center_crop(im_sdr, center_crop_HW)
                    cv2.imwrite(str(im_sdr_export_path), im_sdr)
                    print(blue_text('SDR image exported to: %s'%(str(im_sdr_export_path))))
                
                '''
                export gt depth
                '''
                if modality == 'mi_depth':
                    assert self.os.pts_from['mi']
                    mi_depth_vis_export_path = frame_export_path / ('depth_gt.png' if not if_no_gt_appendix else 'depth.png')
                    mi_depth = self.os.mi_depth_list[frame_idx].squeeze()
                    
                    prediction = mi_depth.copy()
                    prediction[self.os.mi_invalid_depth_mask_list[frame_idx].squeeze()] = np.median(prediction[~self.os.mi_invalid_depth_mask_list[frame_idx].squeeze()])

 
                    prediction_npy_export_path = frame_export_path / ('depth_gt.npy' if not if_no_gt_appendix else 'depth.npy')
                    prediction = center_crop(prediction, center_crop_HW)
                    np.save(str(prediction_npy_export_path), prediction)
                    print(blue_text('depth (npy) exported to: %s'%(str(prediction_npy_export_path))))
                    print('+++ max, min', np.amax(prediction), np.amin(prediction))
                    
                    # export vis of normalized depth
                    depth_vis_normalized, depth_min_and_scale = vis_disp_colormap(mi_depth, normalize=True, valid_mask=~self.os.mi_invalid_depth_mask_list[frame_idx].squeeze())
                    depth_vis_normalized = center_crop(depth_vis_normalized, center_crop_HW)
                    cv2.imwrite(str(mi_depth_vis_export_path), depth_vis_normalized)
                    print(blue_text('depth (vis) exported to: %s'%(str(mi_depth_vis_export_path))))
                    
                if modality == 'mi_seg':
                    assert self.os.pts_from['mi']
                    
                    '''
                    lampMask_%d.png / windowMask_%d.png
                    '''
                    if lamp_dict != {} or window_dict != {}: # ground truth emitters are available
                        objMask = (self.os.mi_seg_dict_of_lists['obj'][frame_idx].squeeze()).astype(float) # True for indoor
                        objMask = center_crop(objMask, center_crop_HW)
                        objMask = objMask.astype(float).astype(bool)
                        ret = self.os.mi_rays_ret_list[frame_idx].shape
                        
                        for emitter_dict in [lamp_dict, window_dict]:
                        # for emitter_dict in [window_dict]:
                        # for emitter_dict in [lamp_dict]:
                            emitter_count = 0
                            for _, (emitter_id, emitter) in enumerate(emitter_dict.items()):
                                emitter_name = 'lamp' if emitter_id in lamp_dict else 'win'
                                if '+' in emitter_id:
                                    emitter_id_list = emitter_id.split('+')
                                else:
                                    emitter_id_list = [emitter_id]
                                # emitter_intensity = emitter['emitter_prop']['intensity']
                            
                                obj_mask_list = []
                                assert len(ret) == self.os.H*self.os.W
                                for s in tqdm(ret):
                                    if s is not None and s.emitter() is not None:
                                        if s.emitter().id() in emitter_id_list:
                                            obj_mask_list.append(1)
                                        else:
                                            obj_mask_list.append(0)
                                    else:
                                        obj_mask_list.append(0)
                                emitter_mask_bool = np.array(obj_mask_list).reshape(self.os.H, self.os.W)
                                emitter_mask = emitter_mask_bool.astype(np.uint8) * 255
                                # mi_emitter_prop = [s.emitter() for s in ret.shape if s is not None and s.emitter() is not None]
                                if np.sum(emitter_mask.reshape(-1)) == 0: continue
                                emitterMask_export_path = frame_export_path / ('%sMask_%d.png'%(emitter_name, emitter_count)) # 255 for emitter
                                cv2.imwrite(str(emitterMask_export_path), emitter_mask)
                                print(blue_text('%sMask exported to: %s'%(emitter_name, str(emitterMask_export_path))))
                                emitter_count += 1
                                
                                if emitter_name == 'win':
                                    objMask = np.logical_and(objMask, ~emitter_mask_bool)
                    else:
                        assert emitter_thres != 0. # get emitter masks from thresholding; pretending to be lamps
                        im_hdr = self.os.im_hdr_list[frame_idx]
                        emitter_mask = (im_hdr.sum(axis=-1) > emitter_thres).astype(np.uint8) * 255
                        # https://stackoverflow.com/questions/16937158/extracting-connected-objects-from-an-image-in-python
                        # smooth the image (to remove small objects)
                        from scipy import ndimage
                        blur_radius = 1.0
                        threshold = 50
                        imgf = ndimage.gaussian_filter(emitter_mask, blur_radius)
                        # find connected components
                        labeled, nr_objects = ndimage.label(imgf > threshold) 
                        print("Number of objects is {}".format(nr_objects))
                        emitter_count = 0
                        for emitter_count_ in range(1, nr_objects+1):
                            emitter_name = 'lamp'
                            emitter_mask_ = labeled == emitter_count_ # labeled==0 is backgroud blob
                            # if np.sum(emitter_mask_) > 0.3 * self.os.H * self.os.W: continue
                            if np.sum(emitter_mask_) < 30: continue
                            # plt.figure()
                            # plt.imshow(labeled)
                            # plt.colorbar()
                            # plt.show()
                            
                            emitter_mask_ = emitter_mask_.astype(np.uint8) * 255
                            emitterMask_export_path = frame_export_path / ('%sMask_%d.png'%(emitter_name, emitter_count)) # 255 for emitter
                            cv2.imwrite(str(emitterMask_export_path), emitter_mask_)
                            print(blue_text('%sMask exported to: %s'%(emitter_name, str(emitterMask_export_path))))
                            emitter_count += 1
                        
                        objMask = np.ones_like(emitter_mask).astype(bool)
                    
                    '''
                    envMask
                    '''
                    envMask = (objMask).astype(np.uint8) * 255
                    envMask_export_path = frame_export_path / 'envMask.png'
                    cv2.imwrite(str(envMask_export_path), envMask)
                    print(blue_text('envMask exported to: %s'%(str(envMask_export_path))))
                    
                if modality in ['lighting', 'emission'] and not IF_CREATED_EDIT_FOLDER:
                    BRDF_results_path = frame_export_path.parent / BRDF_results_folder
                    BRDF_edited_results_path = frame_export_path.parent / (BRDF_results_folder.replace('BRDFLight', 'EditedBRDFLight'))
                    if not BRDF_results_path.exists():
                        print(red('[WARNING] BRDF_results_path does not exist: %s, skipping lighting export'%str(BRDF_results_path)))
                        continue
                    if BRDF_edited_results_path.exists():
                        shutil.rmtree(BRDF_edited_results_path, ignore_errors=True)
                    shutil.copytree(str(BRDF_results_path), str(BRDF_edited_results_path))
                    
                    INPUT_edited_path = Path(str(frame_export_path).replace('input', 'EditedInput'))
                    if INPUT_edited_path.exists():
                        shutil.rmtree(INPUT_edited_path, ignore_errors=True)
                    shutil.copytree(str(frame_export_path), str(INPUT_edited_path))
                    for _ in INPUT_edited_path.iterdir():
                        if _.stem.startswith('DEBUG_'): _.unlink()
                        
                    TEMP_path = Path(str(frame_export_path).replace('input', 'Temp'))
                    TEMP_path.mkdir(exist_ok=True)

                    IF_CREATED_EDIT_FOLDER = True
                        
                if modality == 'emission':
                    geo_mesh_file_list = [_ for _ in BRDF_results_path.iterdir() if _.suffix in ['.obj', '.ply']]
                    pose = self.os.pose_list[frame_idx]
                    _R, _t = pose[:3, :3], pose[:3, 3:4]

                    est_scene_dict = {
                        'type': 'scene',
                    } # create an empty scene; later fill with estimated emitters

                    # if hasattr(self.os, 'shape_id_dict'):
                    # # assert hasattr(self.os, 'shape_id_dict') # assuming mitsuba scene loaded from dict or single/multiple shapes (not from xml)
                    #     est_scene_dict = {
                    #         'type': 'scene',
                    #         'shape_id': self.os.shape_id_dict, 
                    #     }.copy()
                    # else:
                    #     assert self.scene_shape_file != ''
                    #     est_scene_dict = {
                    #         'type': 'scene',
                    #         Path(self.scene_shape_file).stem: {
                    #             # 'id': , 
                    #             'type': Path(self.scene_shape_file).suffix[1:],
                    #             'filename': str(self.scene_shape_file), 
                    #         }, 
                    #     }

                    geo_mesh_id_list = []
                    rad_pred_list = []
                    
                    
                    '''
                    emission debug vis _. Temp/debug_reproj_emission_est.png: ![](https://i.imgur.com/paBh87M.png)
                    '''
                    plt.figure(figsize=(10, 10))
                    plt.axis('equal')
                    ___ = self.os.im_sdr_list[frame_idx].copy()
                    ___ = ___[:, ::-1]
                    plt.imshow(___)
                    
                    for geo_mesh_file in geo_mesh_file_list: # images/demo_lieccv22_brdf_light_result.png; just to make sure the new lights and room geometry matches the original GT geometry shape
                        '''
                        write down transformed mesh back to original space, and write debugging meshes
                        '''
                        assert ('room_' in geo_mesh_file.stem) or ('Lamp' in geo_mesh_file.stem) or ('Win' in geo_mesh_file.stem), 'Unknown geo_mesh_file: %s'%str(geo_mesh_file)
                        _mesh = trimesh.load(str(geo_mesh_file))
                        # transform from opengl cam to opencv world
                        # _light_vertices_cam = (np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]) @ _R.T @ (_light_vertices.T  - _t)).T # convert to oopengl camera coords
                        _vertices_cam = _mesh.vertices
                        if depth_scale_mul is not None:
                            _vertices_cam = _vertices_cam / depth_scale_mul
                            
                        _vertices_cam_rui, (xx_rui, yy_rui) = x_cam_zq_2_x_cam_rui(self.os.K, np.array(_vertices_cam))
                        
                        _vertices_cam_opencv = (np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]) @ _vertices_cam_rui.T).T
                        # _vertices_cam_opencv = _vertices_cam_rui
                        _vertices_cam_world = (_R @ _vertices_cam_opencv.T + _t).T
                        _mesh_world = trimesh.Trimesh(vertices=_vertices_cam_world, faces=_mesh.faces if hasattr(_mesh, 'faces') else None)
                        geo_mesh_file_world = TEMP_path / geo_mesh_file.name
                        _mesh_world.export(str(geo_mesh_file_world))
                        print('[DEBUG] geo_mesh_file_world -> : %s'%str(geo_mesh_file_world))
                        
                        plt.scatter(xx_rui[::-1], yy_rui[::-1], s=1) # debug
                        
                        if 'room' in geo_mesh_file.stem:
                            continue
                        
                        '''
                        construct mitsuba scene in original space with original poses
                        '''
                        shape_id_dict = {
                            'id': geo_mesh_file.stem, 
                            'type': geo_mesh_file.suffix[1:],
                            'filename': str(geo_mesh_file_world), 
                            }
                        geo_mesh_id_list.append(geo_mesh_file.stem)
                        est_scene_dict[geo_mesh_file.stem] = shape_id_dict
                        
                        _emitter_dat_path_list = [_ for _ in BRDF_results_path.iterdir() if _.suffix == '.dat']
                        # _.stem == geo_mesh_file.stem.replace('Pred_0_0', 'Src').replace('Pred_0', 'Src').replace('Pred', 'Src') and 
                        print('----')
                        IF_FOUND = False
                        for _emitter_dat_idx, _emitter_dat_path in enumerate(_emitter_dat_path_list):
                            _emitter_dat_stem = _emitter_dat_path.stem
                            if len(_emitter_dat_stem.split('_')) == 2:
                                _emitter_dat_stem_name, _emitter_dat_stem_idx = _emitter_dat_stem.split('_')
                                geo_mesh_id_match = '%s_0_%d'%(_emitter_dat_stem_name.replace('Src', 'Pred'), int(_emitter_dat_stem_idx))
                            else:
                                _emitter_dat_stem_name = _emitter_dat_stem; _emitter_dat_stem_idx = None
                                geo_mesh_id_match = '%s_0_0'%(_emitter_dat_stem_name.replace('Src', 'Pred'))
                            if geo_mesh_id_match != geo_mesh_file.stem:
                                print('NOT found: ', _emitter_dat_stem_name, _emitter_dat_stem_idx, geo_mesh_id_match, geo_mesh_file.stem)
                                continue
                            print('MATCH: ', _emitter_dat_stem_name, _emitter_dat_stem_idx, geo_mesh_id_match, geo_mesh_file.stem)
                            IF_FOUND = True
                            with open(str(_emitter_dat_path), 'rb') as f:
                                emitter_est_dict = pickle.load(f)
                                # if 'srcSky' in emitter_est_dict:
                                #     _rad_pred = emitter_est_dict['srcSky']
                                _rad_pred = emitter_est_dict['src'].flatten()[:3] # take the sun intensity for windows # [TODO] better way?
                                rad_pred_list.append(_rad_pred)
                        # if len(_emitter_dat_path_list) != 1: # more than one .dat file matches the emitter; dig into this
                        if not IF_FOUND:
                            print('[DEBUG] -----')
                            print(geo_mesh_file.name)
                            print([_.stem for _ in BRDF_results_path.iterdir() if _.suffix == '.dat'])
                            import ipdb; ipdb.set_trace()

                    plt.grid()
                    plt.xlim([0.5, self.os.W-0.5])
                    plt.ylim([0.5, self.os.H-0.5])
                    plt.gca().invert_yaxis()
                    # plt.gca().invert_xaxis() 
                    # plt.show()
                    plt.savefig(str(TEMP_path / 'debug_reproj_emission_est.png'))
                            
                    mi_scene_BRDF_results = mi.load_dict(est_scene_dict)
                    rays_o, rays_d, ray_d_center = self.os.cam_rays_list[frame_idx]
                    rays_o_flatten, rays_d_flatten = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
                    xs_mi = mi.Point3f(self.os.to_d(rays_o_flatten))
                    ds_mi = mi.Vector3f(self.os.to_d(rays_d_flatten))
                    rays_mi = mi.Ray3f(xs_mi, ds_mi)
                    ret = mi_scene_BRDF_results.ray_intersect(rays_mi) # [mitsuba.Scene.ray_intersect] https://mitsuba.readthedocs.io/en/stable/src/api_reference.html?highlight=write_ply#mitsuba.Scene.ray_intersect
                    
                    assert len(ret.shape) == self.os.H*self.os.W
                    
                    emission_mask_list = []
                    for s in tqdm(ret.shape):
                        if s is not None:
                            if s.id() in geo_mesh_id_list:
                                _emitter_idx = geo_mesh_id_list.index(s.id()) + 1
                                emission_mask_list.append(_emitter_idx)
                            else:
                                emission_mask_list.append(0)
                        else:
                            emission_mask_list.append(0)
                    emission_id_map = np.array(emission_mask_list).reshape(self.os.H, self.os.W).astype(np.uint8) # 0 is no emitter

                    emission_mask = (emission_id_map != 0).astype(np.uint8) * 255
                    emission_mask_export_path = INPUT_edited_path / 'emission_mask.png'
                    cv2.imwrite(str(emission_mask_export_path), emission_mask)
                    print(blue_text('emission_mask exported to: %s'%(str(emission_mask_export_path))))
                    
                    emission = np.zeros((emission_mask.shape[0], emission_mask.shape[1], 3), dtype=np.float32)
                    emission_mask_export_path = INPUT_edited_path / 'emission.exr'
                    for _emitter_idx in np.unique(emission_id_map):
                        if _emitter_idx == 0: continue
                        emission[emission_id_map==_emitter_idx, :] = rad_pred_list[_emitter_idx-1].reshape((1, 3)).astype(np.float32)
                    cv2.imwrite(str(emission_mask_export_path), emission)
                    
                if modality == 'lighting':
                    outLight_file_list = [_ for _ in self.os.scene_path.iterdir() if _.stem.startswith('outLight')]
                    assert len(outLight_file_list) > 0, 'No outLight files found at %s'%str(self.os.scene_path)
                    print(white_blue('Found %d outLight files at'%len(outLight_file_list)), str(self.os.scene_path))
                    # assert len(outLight_file_list) == 1
                    
                    IF_light_visible = False
                    outLight_file_count = 0
    
                    # clean all existing light - .dat files in EditedInput
                    for _ in BRDF_edited_results_path.iterdir():
                        if 'visLampSrc_' in str(_):
                            _.unlink(); print('--Unlinked %s'%str(_))

                    # clean all existing lamps files in EditedInput
                    for _ in INPUT_edited_path.iterdir():
                        if 'lampMask' in str(_.stem):
                            _.unlink(missing_ok=True); print('-Unlinked %s'%str(_))
                            
                    '''
                    debug: show the light mask overlayed on the image -> Temp/debug_reproj_new_lights.png ![](https://i.imgur.com/iMHVnlI.png)
                    '''
                    plt.figure(figsize=(15, 5))
                    plt.subplot(131)
                    plt.imshow(self.os.im_sdr_list[frame_idx])
                    light_mask_total = np.empty((self.os.H, self.os.W), dtype=bool)
                    light_mask_total.fill(False)
                        
                    for outLight_file_idx, outLight_file in tqdm(enumerate(outLight_file_list)):
                        root = get_XML_root(str(outLight_file))
                        root = copy.deepcopy(root)
                        shapes = root.findall('shape')
                        # assert len(shapes) > 0, 'No shapes found in %s; double check you XML file (e.g. did you miss headings)?'%str(outLight_file)
                        assert len(shapes) == 1 # one new lamp each XML file for now
                        
                        # for shape in tqdm(shapes):
                        shape = shapes[0]
                        emitters = shape.findall('emitter')
                        assert len(emitters) == 1
                        # if shape.get('type') != 'obj':
                        assert shape.get('type') in ['obj', 'rectangle', 'sphere']
                        #     assert shape.get('type') == 'rectangle'
                            
                        transform_item = shape.findall('transform')[0]
                        transform_accu = np.eye(4, dtype=np.float32)
                        
                        if len(transform_item.findall('rotate')) > 0:
                            rotate_item = transform_item.findall('rotate')[0]
                            _r_h = xml_rotation_to_matrix_homo(rotate_item)
                            transform_accu = _r_h @ transform_accu
                            
                        if len(transform_item.findall('matrix')) > 0:
                            _transform = [_ for _ in transform_item.findall('matrix')[0].get('value').split(' ') if _ != '']
                            transform_matrix = np.array(_transform).reshape(4, 4).astype(np.float32)
                            transform_accu = transform_matrix @ transform_accu # [[R,t], [0,0,0,1]]
                        
                        # assert self.os.extra_transform is None
                        # transform_accu = self.os.extra_transform_homo @ transform_accu
                        if self.os._if_T and not self.os.CONF.scene_params_dict.if_reorient_y_up_skip_shape:
                            # assert hasattr(self.os, 'reorient_transform')
                            transform_accu = self.os._T_homo @ transform_accu
                            
                        transform_item.findall('matrix')[0].set('value', ' '.join(['%.4f'%_ for _ in transform_accu.flatten().tolist()]))
                        xmlString = transformToXml(root)
                        outLight_transformed_file = scene_export_path / 'lightings' / (outLight_file.name.replace('.xml', '_transformed.xml'))
                        (scene_export_path / outLight_transformed_file).parent.mkdir(parents=True, exist_ok=True)
                        with open(str(outLight_transformed_file), 'w') as xmlOut:
                            xmlOut.write(xmlString)
                        
                        # '''
                        # debug: ALTERNATIVELY, dump all lamps as obj, and load obj into xml
                        # in this way, we can additionally directly modify the vertices, thus being able to transform the vertices from Rui to Zhengqin's, which is no longer limited to a rigid transformation
                        # '''
                        # root_obj = copy.deepcopy(root)
                        # if shape.get('type') == 'rectangle':
                        #     (_light_vertices, _light_faces) = get_rectangle_mesh(np.eye(3), np.zeros((3, 1))) # world
                        #     _light_faces -= 1
                        # elif shape.get('type') == 'obj':
                        #     light_mesh_path = self.os.scene_path / shape.findall('string')[0].get('value')
                        #     assert light_mesh_path.exists()
                        #     light_mesh = trimesh.load_mesh(str(light_mesh_path))
                        #     _light_vertices, _light_faces = light_mesh.vertices, light_mesh.faces
                        # elif shape.get('type') == 'sphere':
                        #     light_mesh = trimesh.creation.icosphere(subdivisions=3)
                        #     _light_vertices, _light_faces = light_mesh.vertices, light_mesh.faces
                        
                        # assert np.min(_light_faces) == 0

                        # light_trimesh = trimesh.Trimesh(vertices=_light_vertices, faces=_light_faces)
                        # light_mesh_path = TEMP_path / (shape.get('id')+'.obj')
                        # light_trimesh.export(str(light_mesh_path))
                        # root_obj.findall('shape')[0].set('type', 'obj')
                        # light_mesh_node = et.Element("string", {"name": "filename", "value": str(light_mesh_path)})
                        # root_obj.findall('shape')[0].append(light_mesh_node)
                        # xmlString_obj = transformToXml(root_obj)
                        # outLight_transformed_file_obj = scene_export_path / 'lightings' / (outLight_file.name.replace('.xml', '_transformed_using_obj.xml'))
                        # with open(str(outLight_transformed_file_obj), 'w') as xmlOut:
                        #     xmlOut.write(xmlString_obj)

                        '''
                        dump new light - mask; render with Mitsuba under Rui's world coordinates
                        '''
                        mi_light_scene = mi.load_file(str(outLight_transformed_file))
                        # mi_light_scene = mi.load_file(str(outLight_transformed_file_obj))
                        rays_o, rays_d, ray_d_center = self.os.cam_rays_list[frame_idx]
                        rays_o_flatten, rays_d_flatten = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
                        xs_mi = mi.Point3f(self.os.to_d(rays_o_flatten))
                        ds_mi = mi.Vector3f(self.os.to_d(rays_d_flatten))
                        rays_mi = mi.Ray3f(xs_mi, ds_mi)
                        ret = mi_light_scene.ray_intersect(rays_mi) # [mitsuba.Scene.ray_intersect] https://mitsuba.readthedocs.io/en/stable/src/api_reference.html?highlight=write_ply#mitsuba.Scene.ray_intersect
                        rays_t = ret.t.numpy()
                        rays_v_flatten = rays_t[:, np.newaxis] * rays_d_flatten
                        mi_light_depth = np.sum(rays_v_flatten.reshape(self.os._H(frame_idx), self.os._W(frame_idx), 3) * ray_d_center.reshape(1, 1, 3), axis=-1)
                        invalid_light_depth_mask = np.logical_or(np.isnan(mi_light_depth), np.isinf(mi_light_depth))
                        
                        # print(green_text('=========='), outLight_file_idx, np.sum(~invalid_light_depth_mask))
                        
                        if np.sum(~invalid_light_depth_mask) > 0:
                            IF_light_visible = True
                            
                        if IF_light_visible:
                            # dump new vis lamp mask
                            light_mask_path = INPUT_edited_path / ('lampMask_%d.png'%outLight_file_count)
                            light_mask = (~invalid_light_depth_mask).astype(np.uint8) * 255
                            if len(light_mask.shape) == 3: light_mask = light_mask[:, :, 0]
                            cv2.imwrite(str(light_mask_path), light_mask)
                            
                            light_mask_total = np.logical_or(light_mask_total, light_mask)
                            
                        '''
                        shape or dat; dump to EditedInput/*.obj, in ZQ's coordinates (i.e. camera coordinates and with a different perspective)
                        Loaded together with BRDFLight_../*.obj, should see room goemetry + est lamps + new lamps, in ZQ's coordinates
                        '''
                        if shape.get('type') == 'rectangle':
                            if IF_light_visible:
                                (_light_vertices, _light_faces) = get_rectangle_mesh(transform_accu[:3, :3], transform_accu[:3, 3:4])
                                _light_faces -= 1
                            else:
                                _axes, _axes_center = get_rectangle_thin_box(transform_accu[:3, :3], transform_accu[:3, 3:4]) # (3, 3), each row is a axis
                            # light_trimesh = trimesh.Trimesh(vertices=_light_vertices, faces=_light_faces-1)
                        elif shape.get('type') == 'obj':
                            light_mesh_path = self.os.scene_path / shape.findall('string')[0].get('value')
                            assert light_mesh_path.exists()
                            light_mesh = trimesh.load_mesh(str(light_mesh_path))
                            _light_vertices, _light_faces = light_mesh.vertices, light_mesh.faces
                            if IF_light_visible:
                                _light_vertices = (transform_accu[:3, :3] @ _light_vertices.T + transform_accu[:3, 3:4]).T
                                # light_trimesh = trimesh.Trimesh(vertices=_light_vertices, faces=_light_faces)
                            else:
                                # from lib.utils_OR.utils_OR_mesh import computeBox
                                # [TODO] assuming right now it's axis aligned; need to fix this
                                _axes = transform_accu[:3, :3] @ np.array([
                                    [1., 0., 0.],
                                    [0., 1., 0.],
                                    [0., 0., 1.],
                                ]) *  np.array([
                                    max(np.amax(_light_vertices[:, 0]) - np.amin(_light_vertices[:, 0]), 0.01) / 2., 
                                    max(np.amax(_light_vertices[:, 1]) - np.amin(_light_vertices[:, 1]), 0.01) / 2., 
                                    max(np.amax(_light_vertices[:, 2]) - np.amin(_light_vertices[:, 2]), 0.01) / 2., 
                                ]).reshape((1, 3))
                                _light_vertices = (transform_accu[:3, :3] @ _light_vertices.T + transform_accu[:3, 3:4]).T
                                _axes_center = np.mean(_light_vertices, axis=0).reshape((1, 3))
                        elif shape.get('type') == 'sphere':
                            light_mesh = trimesh.creation.icosphere(subdivisions=3)
                            _light_vertices, _light_faces = light_mesh.vertices, light_mesh.faces
                            if IF_light_visible:
                                _light_vertices = (transform_accu[:3, :3] @ _light_vertices.T + transform_accu[:3, 3:4]).T
                                # light_trimesh = trimesh.Trimesh(vertices=_light_vertices, faces=_light_faces)
                                # light_trimesh.export(str('tmp_light.obj'))
                            else:
                                _axes = transform_accu[:3, :3] @ np.array([
                                    [1., 0., 0.],
                                    [0., 1., 0.],
                                    [0., 0., 1.],
                                ]) *  np.array([
                                    max(np.amax(_light_vertices[:, 0]) - np.amin(_light_vertices[:, 0]), 0.01) / 2., 
                                    max(np.amax(_light_vertices[:, 1]) - np.amin(_light_vertices[:, 1]), 0.01) / 2., 
                                    max(np.amax(_light_vertices[:, 2]) - np.amin(_light_vertices[:, 2]), 0.01) / 2., 
                                ]).reshape((1, 3))
                                _light_vertices = (transform_accu[:3, :3] @ _light_vertices.T + transform_accu[:3, 3:4]).T
                                _axes_center = np.mean(_light_vertices, axis=0).reshape((1, 3))

                        '''
                        dump new light - .obj ∫meshes + .dat; to ZQ's coordinates; NOT a rigid transformation
                        -> when loading room mesh + est lamp meshes from BRDFLight.../*.obj, + new lamp meshes from EditedInput/*.obj, the old and new lamps should roughly overlap:
                        
                        (classroom-275: one new lamp highlighted) ![](https://i.imgur.com/PiwKICQ.jpg)
                        
                        '''
                        
                        pose = self.os.pose_list[frame_idx]
                        _R, _t = pose[:3, :3], pose[:3, 3:4]

                        assert len(shape.findall('emitter')) == 1
                        assert shape.findall('emitter')[0].get('type') == 'area'
                        _rad_item = shape.findall('emitter')[0].findall('rgb')[0]
                        assert _rad_item.get('name') == 'radiance'
                        _rad = [float(_) for _ in _rad_item.get('value').split(',')]
                        assert len(_rad) == 3
                        _rad_max = max(_rad)
                        _rad_new = np.array([_rad[0]/_rad_max, _rad[1]/_rad_max, _rad[2]/_rad_max])
                        _rad_new_BGR = (_rad_new * 255.).astype(np.uint8)[::-1]
                        
                        light_edit_txt_path = INPUT_edited_path / ('light_%d_params.txt'%outLight_file_count)
                        
                        if IF_light_visible:
                            if depth_scale_mul is not None:
                                _light_vertices = ((_light_vertices.T  - _t) * depth_scale_mul + _t).T

                            # _light_vertices_cam = (np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]) @ _R.T @ (_light_vertices.T  - _t)).T # convert to oopengl camera coords
                            
                            _light_vertices_cam = (_R.T @ (_light_vertices.T  - _t)).T
                            _light_vertices_cam = x_cam_rui_2_x_cam_zq(self.os.K, _light_vertices_cam)[0]
                            
                            light_trimesh = trimesh.Trimesh(vertices=_light_vertices_cam, faces=_light_faces)
                            light_mesh_path = INPUT_edited_path / ('visLamp_%d.obj'%outLight_file_count)
                            light_trimesh.export(str(light_mesh_path))

                            # dump new light - .dat
                            light_dat_path = BRDF_edited_results_path / ('visLampSrc_%d.dat'%outLight_file_count)
                            light_dat_dict = {'center': np.mean(_light_vertices_cam, axis=0).reshape((1, 3)), 'src': np.array(_rad).reshape((1, 3))}
                            with open(light_dat_path, 'wb') as fOut:
                                pickle.dump(light_dat_dict, fOut)
                                
                            outLight_file_count += 1
                        else:
                            '''
                            given lieccv22 uses linear transformation on inverse depth, hence non-linear on metric depth, thus the scene is no longer orthographic
                            Then the best thing we can do about invisible lamps is to maintain the geometry as a box in the new scene, and offset it to the correct location
                            '''
                            if depth_scale_mul is not None:
                                _axes_center = ((_axes_center.T  - _t) * depth_scale_mul + _t).T

                            _axes_center_cam = (np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]) @ _R.T @ (_axes_center.T  - _t)).T # convert to oopengl camera coords
                            
                            _axes_cam = (np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]) @ _R.T @ (_axes.T)).T # does not shift or sheer
                            
                            # validation to make sure the axes are orthogonal (after scaling back)
                            _axes_cam_normalized = _axes_cam / np.linalg.norm(_axes_cam, axis=1, keepdims=True)
                            assert np.allclose(_axes_cam_normalized @ _axes_cam_normalized.T, np.eye(3, dtype=np.float32), atol=1e-3)
                            
                            _axes_center_cam_new = _axes_center_cam
                            
                            from lib.utils_lieccv22 import generateBox
                            _light_vertices_cam_new, _light_faces = generateBox(
                                _axes_cam,
                                _axes_center_cam_new.reshape(3,), 
                                0)

                            light_trimesh = trimesh.Trimesh(vertices=_light_vertices_cam_new, faces=_light_faces)
                            light_mesh_path = INPUT_edited_path / ('DEBUG_invLamp_%d.obj'%outLight_file_count)
                            light_trimesh.export(str(light_mesh_path))
                            # clean all existing invlamps files in BRDF_edited_results_path
                            for _ in BRDF_edited_results_path.iterdir():
                                if 'invLamp' in str(_.stem):
                                    _.unlink(missing_ok=True); print('Unlinked %s'%str(_))
                            light_mesh_path = BRDF_edited_results_path / ('invLampPred_0_%d.obj'%outLight_file_count)
                            light_trimesh.export(str(light_mesh_path))
                            
                            # assert outLight_file_count == 0, 'only one light is supported for now'
                            if outLight_file_count > 0:
                                print(yellow('WARNING: only one invis light is supported for now, skipping...'))
                                continue
                            light_dat_path = BRDF_edited_results_path / 'invLampSrc.dat'
                            light_dat_dict = {
                                'center': np.mean(_light_vertices_cam_new, axis=0).reshape((1, 3)).astype(np.float32), 
                                'src': np.array(_rad).reshape((1, 3)).astype(np.float32), 
                                'axes': _axes_cam[np.newaxis].astype(np.float32), 
                                }
                            with open(light_dat_path, 'wb') as fOut:
                                pickle.dump(light_dat_dict, fOut)

                        print(blue_text('NEW light mesh exported to:'), str(light_mesh_path)) # input/visLamp_%d.obj
                        print(blue_text('NEW light .dat exported to:'), str(light_dat_path), light_dat_dict) # {BRDF}/visLampSrc_%d.dat
                        
                        with open(light_edit_txt_path, 'w') as fOut:
                            fOut.write('--ifInvisWin False\n--ifVisWin False\n')
                            if IF_light_visible:
                                fOut.write('--ifInvisLamp False\n--ifVisLamp True\n--isVisLampMesh\n')
                            else:
                                fOut.write('--ifInvisLamp True\n--ifVisLamp False\n')
                        
                        '''
                        dump new light - albedo, im
                        '''
                        if IF_light_visible:
                            albedo_path = BRDF_edited_results_path / 'albedo.npy'
                            albedo = np.load(str(albedo_path)) # (1, 3, H, W)
                            albedo[:, :, light_mask==255] = 0
                            np.save(str(albedo_path), albedo)
                            albedo_im = cv2.imread(str(albedo_path).replace('.npy', '.png'))
                            albedo_im[light_mask==255] = _rad_new_BGR
                            cv2.imwrite(str(albedo_path).replace('.npy', '.png'), albedo_im)
                            _H, _W = albedo_im.shape[:2]
                            albedoDS_im = cv2.resize(albedo_im, (_W//2, _H//2), interpolation=cv2.INTER_AREA)
                            cv2.imwrite(str(albedo_path).replace('.npy', 'DS.png'), albedoDS_im)
                            np.save(str(albedo_path).replace('.npy', 'DS.npy'), albedoDS_im.transpose(2, 0, 1)[np.newaxis])

                            im_path = BRDF_edited_results_path / 'im.png'
                            im = cv2.imread(str(im_path))
                            im[light_mask==255] = 255
                            cv2.imwrite(str(im_path).replace('.png', '.png'), im)
                            im_npy = np.load(str(im_path).replace('.png', '.npy'))
                            im_npy[:, :, light_mask==255] = 0
                            np.save(str(im_path).replace('.png', '.npy'), im_npy)
                            # plt.figure()
                            # plt.subplot(121)
                            # plt.imshow(im[:, :, ::-1])
                            # plt.subplot(122)
                            # plt.imshow(im_npy.squeeze().transpose(1, 2, 0)) # somehow darker...
                            # plt.show()
                            
                            # imSmall_path = '/Volumes/RuiT7/ICCV23/indoor_synthetic/EXPORT_lieccv22/Example1_addLamp_turnOffPredLamps/BRDFLight_size0.200_int0.001_dir1.000_lam0.001_ren1.000_visWin120000_visLamp119540_invWin200000_invLamp150000_optimize/imSmall.png'
                            imSmall_path = BRDF_edited_results_path / 'imSmall.png'
                            imSmall_im = cv2.imread(str(imSmall_path)) # uint8, 0 - 255, (H//2, W//2, 3)
                            imSmall_npy = np.load(str(imSmall_path).replace('.png', '.npy')) # float, 0.-1., (1, 3, H//2, W//2)
                            # plt.figure()
                            # plt.subplot(121)
                            # plt.imshow(imSmall_im)
                            # plt.subplot(122)
                            # plt.imshow(imSmall_npy.squeeze().transpose(1, 2, 0)) # somehow darker...
                            # plt.show()
                            _H, _W = im.shape[:2]
                            __ = np.repeat(cv2.resize(light_mask, (_W//2, _H//2), interpolation=cv2.INTER_NEAREST)[:, :, np.newaxis], 3, -1)
                            light_mask_small = cv2.resize(light_mask, (_W//2, _H//2), interpolation=cv2.INTER_NEAREST)
                            imSmall_im[light_mask_small==255] = __[light_mask_small==255]
                            cv2.imwrite(str(imSmall_path).replace('.png', '.png'), imSmall_im)
                            imSmall_npy[:, :, light_mask_small==255] = __.transpose(2, 0, 1)[np.newaxis][:, :, light_mask_small==255]
                            np.save(str(imSmall_path).replace('.png', '.npy'), imSmall_npy)
                            
                            onMask_path = BRDF_edited_results_path / 'onMask.png'
                            onMask_im = cv2.imread(str(onMask_path)) # uint8, 0 - 255, (H//2, W//2, 3)
                            onMask_npy = np.load(str(onMask_path).replace('.png', '.npy')) # float, 0.-1., (1, 1, H//2, W//2)
                            cv2.imwrite(str(onMask_path).replace('.png', '.png'), onMask_im)
                            cv2.imwrite(str(onMask_path).replace('.png', 'Small.png'), cv2.resize(onMask_im, (_W//2, _H//2), interpolation=cv2.INTER_NEAREST))
                            __ = light_mask[np.newaxis, np.newaxis, :, :].astype(np.float32) / 255.
                            assert np.amax(__) <= 1.
                            np.save(str(onMask_path).replace('.png', '.npy'), __)
                            __ = cv2.resize(light_mask, (_W//2, _H//2), interpolation=cv2.INTER_NEAREST)[np.newaxis, np.newaxis, :, :].astype(np.float32) / 255.
                            np.save(str(onMask_path).replace('.png', 'Small.npy'), __)
                            
                            '''
                            dump new light - depth
                            '''
                            prediction_npy_export_path = INPUT_edited_path / ('depth_gt.npy' if not if_no_gt_appendix else 'depth.npy')
                            depth = np.load(str(prediction_npy_export_path))
                            
                            if depth_scale_mul is not None:
                                depth *= depth_scale_mul
                                
                            depth[light_mask==255] == mi_light_depth
                            depth_export_path = BRDF_edited_results_path / ('depth_gt.npy' if not if_no_gt_appendix else 'depth.npy')
                            np.save(str(depth_export_path), depth)
                            np.save(str(depth_export_path).replace('.npy', 'DS.npy'), depth[::2, ::2][np.newaxis, np.newaxis].astype(np.float32))
                            depth_vis_normalized, depth_min_and_scale = vis_disp_colormap(depth, normalize=True, valid_mask=np.logical_or(~self.os.mi_invalid_depth_mask_list[frame_idx].squeeze(), light_mask==255))
                            depth_vis_normalized = center_crop(depth_vis_normalized, center_crop_HW)
                            cv2.imwrite(str(depth_export_path).replace('.npy', '.png'), depth_vis_normalized)
                            cv2.imwrite(str(depth_export_path).replace('.npy', 'DS.png'), cv2.resize(depth_vis_normalized, (_W//2, _H//2), interpolation=cv2.INTER_AREA))
                            
                            '''
                            dump new light - normal, rough
                            '''
                            mi_light_normal_global = ret.n.numpy().reshape(self.os._H(frame_idx), self.os._W(frame_idx), 3)
                            # FLIP inverted normals!
                            normals_flip_mask = np.logical_and(np.sum(rays_d * mi_light_normal_global, axis=-1) > 0, np.any(mi_light_normal_global != np.inf, axis=-1))
                            if np.sum(normals_flip_mask) > 0:
                                mi_light_normal_global[normals_flip_mask] = -mi_light_normal_global[normals_flip_mask]
                                print(yellow('[mi_sample_rays_pts] %d normals flipped!'%np.sum(normals_flip_mask)))
                            mi_light_normal_global[invalid_light_depth_mask, :] = 0.
                            mi_light_normal_cam_opencv = mi_light_normal_global @ self.os.pose_list[frame_idx][:3, :3]
                            mi_light_normal_cam_opengl = np.stack([mi_light_normal_cam_opencv[:, :, 0], -mi_light_normal_cam_opencv[:, :, 1], -mi_light_normal_cam_opencv[:, :, 2]], axis=-1) # transform normals from OpenGL convention (right-up-backward) to OpenCV (right-down-forward)
                            
                            normal_path = BRDF_edited_results_path / 'normal.npy'
                            normal = np.load(str(normal_path))
                            normal[:, :, light_mask==255] = mi_light_normal_cam_opengl.transpose(2, 0, 1)[np.newaxis][:, :, light_mask==255]
                            normal_vis = np.clip(normal.squeeze().transpose(1, 2, 0)/2.+0.5, 0., 1.)
                            np.save(str(normal_path), normal)
                            np.save(str(normal_path).replace('.npy', 'DS.npy'), normal[:, :, ::2, ::2])
                            cv2.imwrite(str(normal_path).replace('.npy', '.png'), (normal_vis*255).astype(np.uint8))
                            cv2.imwrite(str(normal_path).replace('.npy', 'DS.png'), cv2.resize((normal_vis*255).astype(np.uint8), (_W//2, _H//2), interpolation=cv2.INTER_AREA))
                            
                            rough_path = BRDF_edited_results_path / 'rough.npy'
                            rough = np.load(str(rough_path)) # (1, 1, H, W), 0.-1.
                            rough[:, :, light_mask==255] = 0.
                            np.save(str(rough_path), rough)
                            np.save(str(rough_path).replace('.npy', 'DS.npy'), rough[:, :, ::2, ::2])
                            rough_vis = (rough.squeeze()[:, :, np.newaxis].repeat(3, axis=2)*255).astype(np.uint8)
                            cv2.imwrite(str(rough_path).replace('.npy', '.png'), rough_vis)
                            cv2.imwrite(str(rough_path).replace('.npy', 'DS.png'), cv2.resize(rough_vis, (_W//2, _H//2), interpolation=cv2.INTER_AREA))
                            
                            print(blue_text('lighting FILES exported to: %s'%str(BRDF_edited_results_path)))
                            
        
                    plt.subplot(132)
                    plt.imshow(light_mask_total)
                    plt.subplot(133)
                    plt.imshow(self.os.im_sdr_list[frame_idx]*0.5+light_mask_total[:, :, np.newaxis].astype(np.float32)*np.array([1., 0., 0.]).reshape(1, 1, 3)*0.5)
                    # plt.show()
                    plt.savefig(str(TEMP_path / 'debug_reproj_new_lights.png'))


        frame_list_export_path = self.os.dataset_root / 'EXPORT_lieccv22' / split / ('testList_%s.txt'%self.os.scene_name)
        
        with open(str(frame_list_export_path), 'w') as camOut:
            for frame_export_path in frame_export_path_list:
                frame_export_path = (Path('data') / dataset_name / frame_export_path.relative_to(self.os.dataset_root / 'EXPORT_lieccv22')).parent
                camOut.write('%s\n'%(frame_export_path))
        print(white_blue('Exported test list file to: %s'%(str(frame_list_export_path))))