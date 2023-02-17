import numpy as np
import os.path as osp
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from pathlib import Path
from utils_misc import red, yellow
from utils_OR.utils_OR_geo import isect_line_plane_v3
from lib.utils_io import normalize_v

def read_cam_params_OR(camFile):
    assert osp.isfile(str(camFile))
    with open(str(camFile), 'r') as camIn:
    #     camNum = int(camIn.readline().strip() )
        cam_data = camIn.read().splitlines()
    cam_num = int(cam_data[0])
    cam_params = np.array([x.split(' ') for x in cam_data[1:]]).astype(np.float32)
    if not np.any(cam_params): return []
    assert cam_params.shape[0] == cam_num * 3
    cam_params = np.split(cam_params, cam_num, axis=0) # [[origin, lookat, up], ...]
    return cam_params

def read_K_list_OR(K_list_file):
    assert osp.isfile(str(K_list_file))
    with open(str(K_list_file), 'r') as camIn:
        K_data = camIn.read().splitlines()
    K_num = int(K_data[0])
    K_list = np.array([x.split(' ') for x in K_data[1:]]).astype(np.float32)
    if not np.any(K_list): return []
    assert K_list.shape[0] == K_num * 3
    K_list = np.split(K_list, K_num, axis=0) # [[origin, lookat, up], ...]
    return K_list

def dump_cam_params_OR(pose_file_root: Path, origin_lookat_up_mtx_list: list, Rt_list: list=[], cam_params_dict: dict={}, K_list: list=[], frame_num_all: int=-1, appendix='', extra_transform: np.ndarray=None):
    if frame_num_all != -1 and len(origin_lookat_up_mtx_list) != frame_num_all:
        if_write_pose_file = input(red('pose num to write %d is less than total available poses in the scene (%d poses). Still write? [y/n]'%(len(origin_lookat_up_mtx_list), frame_num_all)))
        if if_write_pose_file in ['N', 'n']:
            print('Aborted writing poses to cam.txt.')
            return

    if_overwrite_pose_file = 'Y'
    if not pose_file_root.exists():
        pose_file_root.mkdir(parents=True, exist_ok=True)
    pose_file_write = pose_file_root / ('cam%s.txt'%appendix)
    if pose_file_write.exists():
        if_overwrite_pose_file = input(red('pose file exists: %s (%d poses). Overwrite? [y/n]'%(str(pose_file_write), len(read_cam_params_OR(pose_file_write)))))
        
    if if_overwrite_pose_file in ['Y', 'y']:
        with open(str(pose_file_write), 'w') as camOut:
            camOut.write('%d\n'%len(origin_lookat_up_mtx_list))
            print('Final sampled camera poses: %d'%len(origin_lookat_up_mtx_list))
            for camPose in origin_lookat_up_mtx_list:
                for n in range(0, 3):
                    camOut.write('%.3f %.3f %.3f\n'%(camPose[n, 0], camPose[n, 1], camPose[n, 2]))
        print(yellow('Pose file written to %s (%d poses).'%(pose_file_write, len(origin_lookat_up_mtx_list))))

        if extra_transform is not None:
            assert extra_transform.shape == (3, 3)
            pose_file_write_extra_transform = str(pose_file_write).replace('.txt', '_extra_transform.txt')
            with open(pose_file_write_extra_transform, 'w') as camOut:
                camOut.write('%d\n'%len(origin_lookat_up_mtx_list))
                print('Final sampled camera poses: %d'%len(origin_lookat_up_mtx_list))
                for camPose in origin_lookat_up_mtx_list:
                    for n in range(0, 3):
                        camPose_ = (extra_transform @ camPose[n].reshape(3, 1)).flatten()
                        camOut.write('%.3f %.3f %.3f\n'%(camPose_[0], camPose_[1], camPose_[2]))
            print(yellow('Pose file (extra transform) written to %s (%d poses).'%(pose_file_write_extra_transform, len(origin_lookat_up_mtx_list))))

        if Rt_list != []:
            Rt_file_write = pose_file_root / ('Rt%s.txt'%appendix)
            with open(str(Rt_file_write), 'w') as RtOut:
                RtOut.write('%d\n'%len(Rt_list))
                print('Final sampled camera poses: %d'%len(origin_lookat_up_mtx_list))
                for R, t in Rt_list:
                    Rt = np.hstack((R, t))
                    Rt = np.vstack((Rt, np.array([0., 0., 0., 1]).reshape(1, 4)))
                    for n in range(0, 4):
                        RtOut.write('%.3f %.3f %.3f %.3f\n'%(Rt[n, 0], Rt[n, 1], Rt[n, 2], Rt[n, 3]))
            print(yellow('Pose (Rt) file written to %s (%d poses).'%(Rt_file_write, len(Rt_list))))
        
        cam_params_dict_write = str(pose_file_root / 'cam_params_dict.txt')
        with open(str(cam_params_dict_write), 'w') as camOut:
            for k, v in cam_params_dict.items():
                camOut.write('%s: %s\n'%(k, str(v)))

        if K_list is not None and len(K_list) > 0:
            K_list_file = str(pose_file_root / 'K_list.txt')
            with open(str(K_list_file), 'w') as camOut:
                camOut.write('%d\n'%len(K_list))
                for K in K_list:
                    assert K.shape == (3, 3)
                    for n in range(0, 3):
                        camOut.write('%.3f %.3f %.3f\n'%(K[n, 0], K[n, 1], K[n, 2]))
            print(yellow('K_list file written to %s (%d Ks).'%(K_list_file, len(K_list))))

def normalize(x):
    return x / np.linalg.norm(x)

def project_v(v, cam_R, cam_t, cam_K, if_only_proj_front_v=False, if_return_front_flags=False, if_v_already_transformed=False, extra_transform_matrix=np.eye(3)):
    if if_v_already_transformed:
        v_transformed = v.T
    else:
        v_transformed = cam_R @ v.T + cam_t
    
    v_transformed = (v_transformed.T @ extra_transform_matrix).T
#     print(v_transformed[2:3, :])
    if if_only_proj_front_v:
        v_transformed = v_transformed * (v_transformed[2:3, :] > 0.)
    p = cam_K @ v_transformed
    if not if_return_front_flags:
        return np.vstack([p[0, :]/(p[2, :]+1e-8), p[1, :]/(p[2, :]+1e-8)]).T
    else:
        return np.vstack([p[0, :]/(p[2, :]+1e-8), p[1, :]/(p[2, :]+1e-8)]).T, (v_transformed[2:3, :] > 0.).flatten().tolist()

def project_3d_line(x1x2, cam_R, cam_t, cam_K, cam_center, cam_zaxis, dist_cam_plane=0., if_debug=False, extra_transform_matrix=np.eye(3)):
    '''
    dist_cam_plane: distance of camera plane to camera center (along cam_zaxis)
    '''
    assert len(x1x2.shape)==2 and x1x2.shape[1]==3
    # print(cam_R.shape, x1x2.T.shape, cam_t.shape)
    x1x2_cam = (cam_R @ x1x2.T + cam_t).T @ extra_transform_matrix
    # print(x1x2_cam)
    if if_debug:
        print('--- x1x2_cam', x1x2_cam)
    front_flags = list(x1x2_cam[:, -1] > dist_cam_plane)
    if if_debug:
        print('--- front_flags', front_flags)
    if not all(front_flags):
        if not front_flags[0] and not front_flags[1]:
            if if_debug:
                print('===> SKIPPED x1x2')
            return None
        # x_isect = isect_line_plane_v3(x1x2[0], x1x2[1], cam_center+cam_zaxis*dist_cam_plane, cam_zaxis, epsilon=1e-6)
        # x1x2 = np.vstack((x1x2[front_flags.index(True)].reshape((1, 3)), x_isect.reshape((1, 3))))
        # x1x2_cam = (cam_R @ x1x2.T + cam_t).T @ extra_transform_matrix
        # x1x2_cam[-1, :] = abs(x1x2_cam[-1, :]) # just in case the z dimension is slightly negative

        x_isect = isect_line_plane_v3(x1x2_cam[0], x1x2_cam[1], np.array([0., 0., 0.01]).reshape((3, 1)), np.array([0., 0., 1.]).reshape((3, 1)), epsilon=1e-6)
        x1x2_cam = np.vstack((x1x2_cam[front_flags.index(True)].reshape((1, 3)), x_isect.reshape((1, 3))))
        # x1x2_cam = (cam_R @ x1x2.T + cam_t).T @ extra_transform_matrix
        # x1x2_cam[-1, :] = abs(x1x2_cam[-1, :]) # just in case the z dimension is slightly negative
    if if_debug:
        print('===> x1x2_cam after', x1x2_cam)

    # x1x2_transformed = x1x2_transformed @ extra_transform_matrix
    # print(x1x2_transformed)
    p = cam_K @ x1x2_cam.T
    # if not if_debug:
    return np.vstack([p[0, :]/(p[2, :]+1e-8), p[1, :]/(p[2, :]+1e-8)]).T
    # else:
    #     return np.vstack([p[0, :]/(p[2, :]+1e-8), p[1, :]/(p[2, :]+1e-8)]).T, x1x2

def get_T_local_to_camopengl_np(normal):
    '''
    args:
        normal: (H, W, 3), normalized
    return:
        camx, camy, normal

    '''
    # assert normal.shape[:2] == (self.imHeight, self.imWidth)
    up = np.array([0, 1, 0], dtype=np.float32)[np.newaxis, np.newaxis] # (1, 1, 3)
    camy_proj = np.sum(up * normal, axis=2, keepdims=True) * normal # (H, W, 3)
    cam_y = up - camy_proj
    cam_y = cam_y / (np.linalg.norm(cam_y, axis=2, keepdims=True) + 1e-6) # (H, W, 3)
    cam_x = - np.cross(cam_y, normal, axis=2)
    cam_x = cam_x / (np.linalg.norm(cam_x, axis=2, keepdims=True) + 1e-6) # (H, W, 3)
    T_local_to_camopengl =  np.stack((cam_x, cam_y, normal), axis=-1)# concat as cols: local2cam; (H, W, 3, 3)

    return T_local_to_camopengl

def origin_lookat_up_to_R_t(origin, lookat, up):
    origin = origin.flatten()
    lookat = lookat.flatten()
    up = up.flatten()
    lookatvector = normalize_v(lookat - origin)
    assert np.amax(np.abs(np.dot(lookatvector.flatten(), up.flatten()))) < 2e-3 # two vector should be perpendicular
    t = origin.reshape((3, 1)).astype(np.float32)
    R = np.stack((np.cross(-up, lookatvector), -up, lookatvector), -1).astype(np.float32)

    return (R, t), lookatvector
    
def R_t_to_origin_lookatvector_up(R, t):
    _, __, lookatvector = np.split(R, 3, axis=-1)
    lookatvector = normalize_v(lookatvector)
    up = normalize_v(-__) # (3, 1)
    assert np.abs(np.sum(lookatvector * up)) < 1e-3
    origin = t

    return (origin, lookatvector, up)

# def project_v_homo(v, cam_transformation4x4, cam_K):
#     # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/img30.gif
#     # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node3.html
#     v_homo = np.hstack([v, np.ones((v.shape[0], 1))])
#     cam_K_homo = np.hstack([cam_K, np.zeros((3, 1))])
# #     v_transformed = cam_R @ v.T + cam_t

#     v_transformed = cam_transformation4x4 @ v_homo.T
#     v_transformed_nonhomo = np.vstack([v_transformed[0, :]/v_transformed[3, :], v_transformed[1, :]/v_transformed[3, :], v_transformed[2, :]/v_transformed[3, :]])
# #     print(v_transformed.shape, v_transformed_nonhomo.shape)
#     v_transformed = v_transformed * (v_transformed_nonhomo[2:3, :] > 0.)
#     p = cam_K_homo @ v_transformed
#     return np.vstack([p[0, :]/p[2, :], p[1, :]/p[2, :]]).T

