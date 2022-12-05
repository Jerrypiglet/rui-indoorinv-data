import shutil
import glob
from tqdm import tqdm
import numpy as np
np.set_printoptions(suppress=True)
import imageio

from pathlib import Path
import mitsuba as mi
from lib.utils_misc import blue_text, yellow, red
from lib.utils_io import convert_write_png
from lib.utils_io import normalize_v
from lib.utils_io import resize_intrinsics

class renderer_mitsubaScene_3D():
    '''
    A class used to visualize/render Mitsuba scene in XML format
    '''
    def __init__(
        self, 
        scene_rendering_path: Path, 
        im_params_dict: dict, 
    ):
        self.scene_rendering_path = scene_rendering_path
        self.im_params_dict = im_params_dict
        
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


