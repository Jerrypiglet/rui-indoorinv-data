import shutil
from pathlib import Path
import numpy as np
import trimesh
import cv2
from tqdm import tqdm
from lib.class_replicaScene3D import replicaScene3D
from lib.utils_io import center_crop
from lib.utils_misc import green, white_red, green_text, yellow, yellow_text, white_blue, blue_text, red, vis_disp_colormap

from lib.class_openroomsScene2D import openroomsScene2D
from lib.class_openroomsScene3D import openroomsScene3D
from lib.class_mitsubaScene3D import mitsubaScene3D
from lib.class_monosdfScene3D import monosdfScene3D
from lib.class_freeviewpointScene3D import freeviewpointScene3D
from lib.class_matterportScene3D import matterportScene3D

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
        if_debug_info: bool=False, 
    ):
        
        valid_scene_object_classes = [openroomsScene2D, openroomsScene3D, mitsubaScene3D, monosdfScene3D, freeviewpointScene3D, matterportScene3D, replicaScene3D]
        assert type(scene_object) in valid_scene_object_classes, '[%s] has to take an object of %s!'%(self.__class__.__name__, ' ,'.join([str(_.__name__) for _ in valid_scene_object_classes]))

        self.os = scene_object
        self.if_force = if_force
        self.format = format
        assert self.format in ['monosdf', 'lieccv22', 'fvp'], '[%s] format: %s is not supported!'%(self.__class__.__name__, self.format)
        self.if_debug_info = if_debug_info

        self.modality_list_export = list(set(modality_list))
        for _ in self.modality_list_export:
            if _ == '': continue
            assert _ in self.valid_modalities, '[%s] modality: %s is not supported!'%(self.__class__.__name__, _)

    @property
    def valid_modalities(self):
        return ['im_hdr', 'im_sdr', 'poses', 'im_mask', 'shapes', 'mi_normal', 'mi_depth']
    
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
        
        scene_export_path.mkdir(parents=True, exist_ok=True)
        
        return True
        
    def export_monosdf_fvp(self, modality_list={}, appendix='', split='', format='monosdf'):
        '''
        export scene to mitsubaScene data structure + monosdf inputs
        and fvp: free viewpoint dataset https://gitlab.inria.fr/sibr/projects/indoor_relighting
        '''
        
        assert format in ['monosdf', 'fvp'], 'format %s not supported'%format
        
        scene_export_path = self.os.rendering_root / ('EXPORT_%s'%format) / (self.os.scene_name + appendix)
        if split != '':
            scene_export_path = scene_export_path / split

        if self.prepare_check_export(scene_export_path) == False:
            return

        modality_list_export = modality_list if len(modality_list) > 0 else self.modality_list_export

        for modality in modality_list_export:
            # assert modality in self.modality_list, 'modality %s not in %s'%(modality, self.modality_list)

            if modality == 'poses':
                if format == 'monosdf': 
                    self.os.export_poses_cam_txt(scene_export_path, cam_params_dict=self.os.cam_params_dict, frame_num_all=self.os.frame_num_all)
                    '''
                    cameras for MonoSDF
                    '''
                    cameras = {}
                    scale_mat_path = self.os.rendering_root / 'scene_export' / (self.os.scene_name + appendix) / 'scale_mat.npy'
                    if split != 'val':
                        poses = [np.vstack((pose, np.array([0., 0., 0., 1.], dtype=np.float32).reshape((1, 4)))) for pose in self.os.pose_list]
                        poses = np.array(poses)
                        assert poses.shape[1:] == (4, 4)
                        min_vertices = poses[:, :3, 3].min(axis=0)
                        max_vertices = poses[:, :3, 3].max(axis=0)
                        center = (min_vertices + max_vertices) / 2.
                        scale = 2. / (np.max(max_vertices - min_vertices) + 3.)
                        print('[pose normalization to unit cube] --center, scale--', center, scale)
                        # we should normalized to unit cube
                        scale_mat = np.eye(4).astype(np.float32)
                        scale_mat[:3, 3] = -center
                        scale_mat[:3 ] *= scale 
                        scale_mat = np.linalg.inv(scale_mat)
                        np.save(str(scale_mat_path), {'scale_mat': scale_mat, 'center': center, 'scale': scale})
                    else:
                        assert scale_mat_path.exists(), 'scale_mat.npy not found in %s'%str(scale_mat_path)
                        scale_mat_dict = np.load(str(scale_mat_path), allow_pickle=True).item()
                        scale_mat = scale_mat_dict['scale_mat']
                        center = scale_mat_dict['center']
                        scale = scale_mat_dict['scale']

                    cameras['center'] = center
                    cameras['scale'] = scale
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
                            assert K[0][0] == K[1][1]
                            assert K[0][1] == 0. and K[1][0] == 0. # no redial distortion
                            camOut.write('%.6f %d %d\n'%(f, 0, 0)) # <f> <k1> <k2>   [the focal length, followed by two radial distortion coeffs]
                            
                            # [!!!] assert if_scale_scene = 1.0
                            R, t = self.os.pose_list[frame_idx][:3, :3], self.os.pose_list[frame_idx][:3, 3]
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
                        
                else:
                    raise NotImplementedError

            if modality == 'im_hdr':
                file_str = {'monosdf': 'Image/%03d_0001.exr', 'fvp': 'images/%05d.exr'}[format]
                (scene_export_path / file_str).parent.mkdir(parents=True, exist_ok=True)
                for frame_idx, frame_id in enumerate(self.os.frame_id_list):
                    im_hdr_export_path = scene_export_path / (file_str%frame_idx)
                    hdr_scale = self.os.hdr_scale_list[frame_idx]
                    cv2.imwrite(str(im_hdr_export_path), hdr_scale * self.os.im_hdr_list[frame_idx][:, :, [2, 1, 0]])
                    print(blue_text('HDR image %d exported to: %s'%(frame_id, str(im_hdr_export_path))))
            
            if modality == 'im_sdr':
                file_str = {'monosdf': 'Image/%03d_0001.png', 'fvp': 'images/%05d.jpg'}[format]
                (scene_export_path / file_str).parent.mkdir(parents=True, exist_ok=True)
                for frame_idx, frame_id in enumerate(self.os.frame_id_list):
                    im_sdr_export_path = scene_export_path / (file_str%frame_idx)
                    cv2.imwrite(str(im_sdr_export_path), (np.clip(self.os.im_sdr_list[frame_idx][:, :, [2, 1, 0]], 0., 1.)*255.).astype(np.uint8))
                    print(blue_text('SDR image %d exported to: %s'%(frame_id, str(im_sdr_export_path))))
                    print(self.os.modality_file_list_dict['im_sdr'][frame_idx])
            
            if modality == 'mi_normal':
                assert format == 'monosdf'
                (scene_export_path / 'MiNormalGlobal').mkdir(parents=True, exist_ok=True)
                assert self.os.pts_from['mi']
                for frame_idx, frame_id in enumerate(self.os.frame_id_list):
                    mi_normal_export_path = scene_export_path / 'MiNormalGlobal' / ('%03d_0001.png'%frame_idx)
                    cv2.imwrite(str(mi_normal_export_path), (np.clip(self.os.mi_normal_global_list[frame_idx][:, :, [2, 1, 0]]/2.+0.5, 0., 1.)*255.).astype(np.uint8))
                    print(blue_text('Mitsuba normal (global) %d exported to: %s'%(frame_id, str(mi_normal_export_path))))
            
            if modality == 'mi_depth':
                assert format == 'monosdf'
                (scene_export_path / 'MiDepth').mkdir(parents=True, exist_ok=True)
                assert self.os.pts_from['mi']
                for frame_idx, frame_id in enumerate(self.os.frame_id_list):
                    mi_depth_vis_export_path = scene_export_path / 'MiDepth' / ('%03d_0001.png'%frame_idx)
                    mi_depth = self.os.mi_depth_list[frame_idx].squeeze()
                    depth_normalized, depth_min_and_scale = vis_disp_colormap(mi_depth, normalize=True, valid_mask=~self.os.mi_invalid_depth_mask_list[frame_idx])
                    cv2.imwrite(str(mi_depth_vis_export_path), depth_normalized)
                    print(blue_text('depth (vis) %d exported to: %s'%(frame_id, str(mi_depth_vis_export_path))))
                    mi_depth_npy_export_path = scene_export_path / 'MiDepth' / ('%03d_0001.npy'%frame_idx)
                    np.save(str(mi_depth_npy_export_path), mi_depth)
                    print(blue_text('depth (npy) %d exported to: %s'%(frame_id, str(mi_depth_npy_export_path))))

            if modality == 'im_mask':
                file_str = {'monosdf': 'ImMask/%03d_0001.png', 'fvp': 'images/%08d_mask.png'}[format]
                (scene_export_path / file_str).parent.mkdir(parents=True, exist_ok=True)
                
                assert self.os.pts_from['mi']
                if hasattr(self.os, 'im_mask_list'):
                    assert len(self.os.im_mask_list) == len(self.os.mi_invalid_depth_mask_list)
                if_undist_mask = False
                if hasattr(self.os, 'if_undist'):
                    if self.os.if_undist:
                        assert hasattr(self.os, 'im_undist_mask_list')
                        assert len(self.os.im_undist_mask_list) == len(self.os.mi_invalid_depth_mask_list)
                        if_undist_mask = True
                for frame_idx, frame_id in enumerate(self.os.frame_id_list):
                    im_mask_export_path = scene_export_path / (file_str%frame_idx)
                    mi_invalid_depth_mask = self.os.mi_invalid_depth_mask_list[frame_idx]
                    assert mi_invalid_depth_mask.dtype == bool
                    im_mask_export = ~mi_invalid_depth_mask
                    mask_source_list = ['mi']
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
            
            if modality == 'shapes': 
                assert self.os.if_loaded_shapes, 'shapes not loaded'
                T_list_ = [(None, '')]
                if self.os.extra_transform is not None:
                    T_list_.append((self.extra_transform_inv, '_extra_transform'))

                for T_, appendix in T_list_:
                    shape_list = []
                    # shape_export_path = scene_export_path / ('scene%s.obj'%appendix)
                    file_str = {'monosdf': 'scene%s.obj'%appendix, 'fvp': 'meshes/recon.ply'}[format]
                    (scene_export_path / file_str).parent.mkdir(parents=True, exist_ok=True)
                    shape_export_path = scene_export_path / file_str
                    
                    for vertices, faces in zip(self.os.vertices_list, self.os.faces_list):
                        if T_ is not None:
                            shape_list.append(trimesh.Trimesh((T_ @ vertices.T).T, faces-1))
                        else:
                            shape_list.append(trimesh.Trimesh(vertices, faces-1))
                    shape_tri_mesh = trimesh.util.concatenate(shape_list)
                    shape_tri_mesh.export(str(shape_export_path))
                    if not shape_tri_mesh.is_watertight:
                        trimesh.repair.fill_holes(shape_tri_mesh)
                        shape_tri_mesh_convex = trimesh.convex.convex_hull(shape_tri_mesh)
                        shape_tri_mesh_convex.export(str(shape_export_path.parent / ('%s_hull%s.obj'%(shape_export_path.stem, appendix))))
                        shape_tri_mesh_fixed = trimesh.util.concatenate([shape_tri_mesh, shape_tri_mesh_convex])
                        if format == 'monosdf':
                            shape_tri_mesh_fixed.export(str(shape_export_path.parent / ('%s_fixed%s.obj'%(shape_export_path.stem, appendix))))
                        elif format == 'fvp': # overwrite the original mesh
                            shape_tri_mesh_fixed.export(str(shape_export_path))
                            # scale.txt
                            scene_scale = self.os.scene_scale if hasattr(self.os, 'scene_scale') else 1.
                            with open(str(scene_export_path / 'scale.txt'), 'w') as camOut:
                                camOut.write('%.4f\n'%scene_scale)
                        else:
                            raise NotImplementedError
                        print(yellow('Mesh is not watertight. Filled holes and added convex hull: -> %s%s.obj, %s_hull%s.obj, %s_fixed%s.obj'%(shape_export_path.name, appendix, shape_export_path.name, appendix, shape_export_path.name, appendix)))

    def export_lieccv22(self, modality_list=[], appendix='', split='', center_crop_HW=None, assert_shape=None, window_area_emitter_id_list: list=[], merge_lamp_id_list: list=[], if_no_gt_appendix: bool=False):
        '''
        export lieccv22 scene to Zhengqin's ECCV'22 format
        
        - window_area_emitter_id: id for the are light which is acrually a window...
        - merge_lamp_id_list: list of lamp ids to be merged into one
        
        '''
        scene_export_path = self.os.rendering_root / self.os.scene_name / 'EXPORT_lieccv22' / split
        if self.prepare_check_export(scene_export_path) == False:
            return
        
        modality_list_export = modality_list if len(modality_list) > 0 else self.modality_list_export
        
        if assert_shape is not None:
            assert assert_shape == (self.os.H, self.os.W), 'assert_shape %s not equal to (H, W) %s'%(assert_shape, (self.os.H, self.os.W))
            
        lamp_dict = {_['id'].replace('emitter-', ''): _ for _ in self.os.lamp_list}
        window_dict = {_['id'].replace('emitter-', ''): _ for _ in self.os.window_list}
        if window_area_emitter_id_list != []:
            for window_area_emitter_id in window_area_emitter_id_list:
                assert window_area_emitter_id in lamp_dict
                window_dict.update({window_area_emitter_id: lamp_dict[window_area_emitter_id]})
                lamp_dict.pop(window_area_emitter_id)
        if merge_lamp_id_list != []:
            assert [_ in lamp_dict for _ in merge_lamp_id_list]
            new_lamp = [lamp_dict[_] for _ in merge_lamp_id_list]
            enw_lamp_id = '+'.join(merge_lamp_id_list)
            lamp_dict.update({enw_lamp_id: new_lamp})
            for lamp_id in merge_lamp_id_list:
                lamp_dict.pop(lamp_id)
            
        print(yellow('🛋️ Found %d lamps and %d windows'%(len(lamp_dict), len(window_dict))))
        
        frame_export_path_list = []
        for frame_idx, frame_id in enumerate(self.os.frame_id_list):
            frame_export_path = self.os.rendering_root / 'lieccv22_export' / split / (self.os.scene_name + '_frame%d'%frame_id + appendix) / 'input'
            frame_export_path.mkdir(parents=True, exist_ok=True)
            frame_export_path_list.append(frame_export_path)
            
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
                    depth_normalized, depth_min_and_scale = vis_disp_colormap(mi_depth, normalize=True, valid_mask=~self.os.mi_invalid_depth_mask_list[0])
                    depth_normalized = center_crop(depth_normalized, center_crop_HW)
                    cv2.imwrite(str(mi_depth_vis_export_path), depth_normalized)
                    print(blue_text('depth (vis) exported to: %s'%(str(mi_depth_vis_export_path))))
                    mi_depth_npy_export_path = frame_export_path / ('depth_gt.npy' if not if_no_gt_appendix else 'depth.npy')
                    mi_depth = center_crop(mi_depth, center_crop_HW)
                    np.save(str(mi_depth_npy_export_path), mi_depth)
                    print(blue_text('depth (npy) exported to: %s'%(str(mi_depth_npy_export_path))))
                    
                if modality == 'mi_seg':
                    assert self.os.pts_from['mi']
                    
                    ret = self.os.mi_rays_ret_list[frame_idx].shape
                    objMask = (self.os.mi_seg_dict_of_lists['obj'][frame_idx].squeeze()).astype(float) # True for indoor
                    objMask = center_crop(objMask, center_crop_HW)
                    objMask = objMask.astype(float).astype(bool)
                    
                    '''
                    lampMask_%d.png / windowMask_%d.png
                    '''
                    # for emitter_dict in [lamp_dict, window_dict]:
                    # for emitter_dict in [window_dict]:
                    for emitter_dict in [lamp_dict]:
                        for _, (emitter_id, emitter) in enumerate(emitter_dict.items()):
                            emitter_name = 'lamp' if emitter_id in lamp_dict else 'win'
                            emitter_count = 0
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
                    
                    '''
                    envMask
                    '''
                    envMask = (objMask).astype(np.uint8) * 255
                    envMask_export_path = frame_export_path / 'envMask.png'
                    cv2.imwrite(str(envMask_export_path), envMask)
                    print(blue_text('envMask exported to: %s'%(str(envMask_export_path))))
                        
        frame_list_export_path = self.os.rendering_root / 'lieccv22_export' / split / 'testList.txt'
        with open(str(frame_list_export_path), 'w') as camOut:
            for frame_export_path in frame_export_path_list:
                frame_export_path = (Path('data/kitchen') / frame_export_path.relative_to(self.os.rendering_root / 'lieccv22_export')).parent
                camOut.write('%s\n'%(frame_export_path))
        print(white_blue('Exported test list file to: %s'%(str(frame_list_export_path))))