import sys
sys.path.insert(0, '/Users/jerrypiglet/Documents/Projects/OpenRooms_RAW_loader')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndimage
import lib.utils_OR.utils_OR_indoor_light_editing as utils
import open3d as o3d
from lib.utils_OR.utils_OR_mesh import loadMesh, writeMesh

'''
Zhengqin's direct shading rendering. 
In 2D camera space; requires lamps to be visible.
'''

class renderDirecLighting:
    def __init__(self,
            fov=57.95, isCuda = True, max_plate = 256):
        self.fov = fov / 180.0 * np.pi
        self.isCuda = isCuda
        self.F0 = 0.05
        self.max_plate = max_plate

    def loadMesh(self, visLampMeshName, isCuda=False):
        mesh = o3d.io.read_triangle_mesh(visLampMeshName)
        vertices = np.array( mesh.vertices).astype(np.float32)
        faces = np.array(mesh.triangles).astype(np.int32) # 0-based

        v1 = vertices[faces[:, 0], :]
        v2 = vertices[faces[:, 1], :]
        v3 = vertices[faces[:, 2], :]

        lpts = 1.0 / 3.0 * (v1 + v2 + v3)
        e1 = v2 - v1
        e2 = v3 - v1
        lpts_normal = np.cross(e1, e2)
        lpts_area = 0.5 * np.sqrt(np.sum(
            lpts_normal * lpts_normal, axis=1, keepdims = True))
        lpts_normal = lpts_normal / np.maximum(2 * lpts_area, 1e-6)

        center = np.mean(vertices, axis=0, keepdims = True)

        normal_flip = (np.sum(lpts_normal * (lpts - center), axis=1, keepdims=True) < 0)
        normal_flip = normal_flip.astype(np.float32)
        lpts_normal = -lpts_normal * normal_flip + (1 - normal_flip) * lpts_normal

        plate_num = lpts.shape[0]

        lpts = lpts.transpose(1, 0).reshape(1, 3, 1, plate_num)
        lpts_normal = lpts_normal.transpose(1, 0).reshape(1, 3, 1, plate_num)
        lpts_area = lpts_area.reshape(1, 1, 1, plate_num)

        lpts = torch.from_numpy(lpts)
        lpts_normal = torch.from_numpy(lpts_normal)
        lpts_area = torch.from_numpy(lpts_area)
        if isCuda:
            lpts = lpts.cuda()
            lpts_normal = lpts_normal.cuda()
            lpts_area = lpts_area.cuda()

        if plate_num > self.max_plate:
            prob = float(self.max_plate)  / float(plate_num)
            select_ind = np.random.choice([0, 1], size=(plate_num), p=[1-prob, prob])
            select_ind = torch.from_numpy(select_ind).long()
            if isCuda:
                select_ind = select_ind.cuda()

            lpts = lpts[:, :, :, select_ind == 1]
            lpts_normal = lpts_normal[:, :, :, select_ind == 1]
            lpts_area = lpts_area[:, :, :, select_ind == 1]
        else:
            prob = 1

        return lpts, lpts_normal, lpts_area, prob

    def depthToPoint(self, depth):
        height, width = depth.size(2), depth.size(3)
        xRange = 1 * np.tan(self.fov / 2)
        yRange = float(height) / float(width) * xRange

        x, y = np.meshgrid(np.linspace(-xRange, xRange, width),
                np.linspace(-yRange, yRange, height))

        y = np.flip(y, axis=0)
        z = -np.ones( (height, width), dtype=np.float32)

        pCoord = np.stack([x, y, z], axis = 0)[np.newaxis, :]
        pCoord = pCoord.astype(np.float32)
        pCoord = torch.from_numpy(pCoord)
        if self.isCuda:
            pCoord = pCoord.cuda()
        point = pCoord * depth

        return point

    # def maskToEdge(self, pts_mask):
    #     batchSize, _, height, width = pts_mask.size()
    #     edges, masks  = [], []
    #     for n in range(0, batchSize):
    #         mask = pts_mask[n, 0, :].detach()
    #         if self.isCuda:
    #             mask = mask.cpu().numpy()
    #         else:
    #             mask = mask.numpy()

    #         mask = (mask == 1)
    #         mask_s = ndimage.binary_erosion(mask, structure = np.ones((3, 3)))
    #         edge = mask.astype(np.float32) - mask_s.astype(np.float32)
    #         edge = np.clip(edge, 0, 1)

    #         mask = mask.reshape(1, 1, height, width)
    #         edge = edge.reshape(1, 1, height, width)
    #         mask = torch.from_numpy(mask.astype(np.float32))
    #         edge = torch.from_numpy(edge.astype(np.float32))
    #         if self.isCuda:
    #             mask = mask.cuda()
    #             edge = edge.cuda()

    #         edges.append(edge)
    #         masks.append(mask)

    #     edges = torch.cat(edges, dim=0)
    #     masks = torch.cat(masks, dim=0)

    #     return masks, edges

    # def maskToPlates(self, pts, pts_mask, pts_edge, pts_normal, lpt_center, pixel_size):
    #     # pts: 1 x 3 x N x 1
    #     # pts_mask: 1 x 1 x N x 1
    #     # pts_normal: 1 x 3 x N x 1
    #     # lpt_center: 1 x 3 x 1 x 1

    #     # compute foreground plates
    #     pts_mask = pts_mask.reshape(-1)
    #     pts_edge = pts_edge.reshape(-1)

    #     pNum = pts.size(2)
    #     pts = pts.view(3, pNum)
    #     pts_normal = pts_normal.view(3, pNum)

    #     fg_center = pts[:, pts_mask > 0.9].detach()

    #     edge_sp = pts[:, pts_edge > 0.9].detach()

    #     plateNum = fg_center.size(1)
    #     edgeNum = edge_sp.size(1)

    #     fg_normal = pts_normal[:, pts_mask > 0.9].detach()

    #     fg_area = fg_center[2:3, :] * fg_center[2:3, :] * pixel_size \
    #             / torch.clamp(fg_normal[2:3, :].abs(), min=1e-1)
    #     edge_width = edge_sp[2:3, :].abs() * np.sqrt(pixel_size)

    #     # compute background plates
    #     lpt_center = lpt_center.view(3, 1)
    #     lpt_direc = lpt_center / torch.sqrt(
    #         torch.clamp(
    #             torch.sum(lpt_center * lpt_center, dim=0, keepdim = True),
    #             min=1e-12
    #     )
    # )

    #     bg_center = torch.sum( (fg_center - lpt_center) * lpt_direc,
    #             dim=0, keepdim=True).abs() * 2 * lpt_direc + fg_center
    #     bg_normal = torch.sum(fg_normal * lpt_direc, dim=0, keepdim=True).abs() \
    #             * lpt_direc * 2 + fg_normal

    #     edge_center = torch.sum( (edge_sp - lpt_center) * lpt_direc,
    #             dim=0, keepdim=True).abs() * lpt_direc + edge_sp
    #     edge_normal = edge_center - lpt_center - \
    #             torch.sum( (edge_center - lpt_center) * lpt_direc, dim=0, keepdim=True) * lpt_direc
    #     edge_normal = edge_normal / torch.sqrt(
    #         torch.clamp(
    #             torch.sum(edge_normal * edge_normal, dim=0, keepdim=True),
    #             min=1e-12
    #     )
    # )
    #     edge_len = 2 * torch.sqrt(
    #             torch.clamp(
    #                 torch.sum((edge_center -  edge_sp) * (edge_center - edge_sp), dim=0, keepdim=True),
    #                 min=1e-6
    #             )
    #         )
    #     edge_area = edge_len * edge_width

    #     fg_center = fg_center.view(1, 3, 1, plateNum)
    #     bg_center = bg_center.view(1, 3, 1, plateNum)
    #     edge_center = edge_center.view(1, 3, 1, edgeNum)

    #     fg_normal = fg_normal.view(1, 3, 1, plateNum)
    #     bg_normal = bg_normal.view(1, 3, 1, plateNum)
    #     edge_normal = edge_normal.view(1, 3, 1, edgeNum)

    #     fg_area = fg_area.view(1, 1, 1, plateNum)
    #     edge_area = edge_area.view(1, 1, 1, edgeNum)

    #     plate_center = torch.cat([fg_center, bg_center, edge_center], dim=3)
    #     plate_normal = torch.cat([fg_normal, bg_normal, edge_normal], dim=3)
    #     plate_area = torch.cat([fg_area, fg_area, edge_area], dim=3)

    #     plate_num = plate_center.size(3)
    #     if plate_num > self.max_plate:
    #         prob = float(self.max_plate)  / float(plate_num)
    #         select_ind = np.random.choice([0, 1], size=(plate_num), p=[1-prob, prob])
    #         select_ind = torch.from_numpy(select_ind).long()

    #         plate_center = plate_center[:, :, :, select_ind == 1]
    #         plate_normal = plate_normal[:, :, :, select_ind == 1]
    #         plate_area = plate_area[:, :, :, select_ind == 1]
    #     else:
    #         prob = 1

    #     return plate_center, plate_normal, plate_area, prob

    def forward(
            self,
            lpt_center,
            lpt_int,
            depth,
            pts_mask,
            pts_normal,
            isTest,
            visLampMeshNames = None,
            pts_rough = None):

        lpt_int = lpt_int.unsqueeze(-1).unsqueeze(-1)
        lpt_center = lpt_center.unsqueeze(-1).unsqueeze(-1)
        pts = self.depthToPoint(depth)
        # pts_mask, pts_edge = self.maskToEdge(pts_mask)

        bn = pts.size(0)
        height, width = pts.size(2), pts.size(3)
        # pixel_len = np.tan(self.fov / 2.0) / width * 2
        # pixel_size = pixel_len * pixel_len

        N = width * height
        pts = pts.view(bn, 3, N, 1)
        pts_normal = pts_normal.view(bn, 3, N, 1)
        pts_mask = pts_mask.view(bn, 1, N, 1)
        # pts_edge = pts_edge.view(bn, 1, N, 1)
        if not pts_rough is None:
            pts_rough = pts_rough.view(bn, 1, N, 1)

        # Build the x and y axis for every place
        # Build the x and y axis for every place
        # pts:          batchSize x 3 x N x 1
        # pts_normal:   batchSize x 3 x N x 1

        # lpt_center:   batchSize x 3 x 1 x 1
        # lpt_int:      batchSize x 3 x 1 x 1

        pts_shading_arr  = []
        if not pts_rough is None:
            pts_specular_arr = []
        lpts_arr = []

        for n in range(0, bn):
            if visLampMeshNames is None:
                # lpts, lpts_normal, lpts_area, prob = self.maskToPlates(
                #         pts[n:n+1, :], pts_mask[n:n+1, :], pts_edge[n:n+1, :], pts_normal[n:n+1, :],
                #         lpt_center[n:n+1, :], pixel_size)
                assert False
            else:
                lpts, lpts_normal, lpts_area, prob = self.loadMesh(visLampMeshNames[n])
            lpts_arr.append(lpts.squeeze(2).permute(0, 2, 1))

            pts_dir = lpts - pts[n:n+1, :]
            pts_distL2 = torch.clamp(torch.sum(pts_dir * pts_dir, dim=1, keepdim=True), min=1e-12)
            pts_dir = pts_dir / torch.sqrt(pts_distL2)

            pts_cos = torch.sum(pts_dir * pts_normal[n:n+1, :], dim=1, keepdim=True)
            lpt_cos = torch.clamp(torch.sum(pts_dir * lpts_normal, dim=1, keepdim=True), -1, 1)

            if isTest:
                pts_int = lpt_int[n:n+1, :] * torch.clamp(pts_cos, min=0, max=1) \
                        * torch.clamp(lpt_cos, min=0, max=1)
            else:
                pts_int = lpt_int[n:n+1, :] * torch.clamp(pts_cos, min=0, max=1) \
                        * lpt_cos.abs()
                import ipdb; ipdb.set_trace()

            pts_shading = pts_int / pts_distL2.detach() * lpts_area.detach() / prob

            pts_shading = torch.sum(pts_shading, dim=-1)
            pts_shading = pts_shading.view(1, 3, height, width)
            pts_shading_arr.append(pts_shading)

            if not pts_rough is None:
                alpha = pts_rough * pts_rough
                k = (pts_rough + 1) * (pts_rough + 1) / 8.0
                alpha2 = alpha * alpha

                v = -pts
                v = v / torch.sqrt(torch.clamp(torch.sum(v * v, dim=1), min=1e-6))
                l = pts_dir
                h = (l + v) / 2.0
                h = h / torch.sqrt(torch.clamp(torch.sum(h * h, dim=1, keepdim=True), min=1e-6))
                vdh = torch.sum(v * h, dim=1, keepdim=True)

                temp = (torch.zeros([1, 1, 1, 1, 1], dtype=torch.float32) + 2.0)
                if self.isCuda:
                    temp = temp.cuda()
                frac0 = self.F0 + (1 - self.F0) * torch.pow(temp, (-5.55472*vdh - 6.98326)*vdh)

                ndv = torch.clamp(torch.sum(pts_normal * v, dim=1, keepdim=True), 0, 1)
                ndh = torch.clamp(torch.sum(pts_normal * h, dim=1, keepdim=True), 0, 1)
                ndl = torch.clamp(torch.sum(pts_normal * l, dim=1, keepdim=True), 0, 1) # == pts_cos

                frac = alpha2 * frac0
                nom0 = ndh * ndh * (alpha2 -1) + 1
                nom1 = ndv * (1 - k) + k
                nom2 = ndl * (1 - k) + k
                nom = torch.clamp(4*np.pi * nom0*nom0*nom1*nom2, 1e-6, 4*np.pi)
                pts_specular = frac / nom * pts_int  / pts_distL2.detach() * lpts_area.detach()
                pts_specular = torch.sum(pts_specular, dim=-1)

                pts_specular = pts_specular.view(1, 3, height, width)
                pts_speclar_arr.append(pts_specular)


        pts_shading = torch.cat(pts_shading_arr, dim=0)
        if not pts_rough is None:
            pts_speculars = torch.cat(pts_speculars, dim=0)
            return pts_shading, pts_specular, lpts_arr
        else:
            return pts_shading, lpts_arr


if __name__ == '__main__':
    import pickle
    import cv2
    import struct
    import scipy.ndimage as ndimage
    from pathlib import Path

    from lib.global_vars import PATH_HOME, OR_RAW_ROOT, host, mi_variant

    import sys
    sys.path.insert(0, PATH_HOME)
    from pathlib import Path
    import numpy as np
    import numpy as np
    np.set_printoptions(suppress=True)
    from lib.class_openroomsScene3D import openroomsScene3D
    
    base_root = Path(PATH_HOME) / 'data/public_re_3'
    xml_root = Path(PATH_HOME) / 'data/public_re_3/scenes'
    # intrinsics_path = Path(PATH_HOME) / 'data/intrinsic.txt'
    semantic_labels_root = Path(PATH_HOME) / 'files_openrooms'
    layout_root = Path(OR_RAW_ROOT) / 'layoutMesh'
    shapes_root = Path(OR_RAW_ROOT) / 'uv_mapped'
    envmaps_root = Path(OR_RAW_ROOT) / 'EnvDataset' # not publicly availale
    shape_pickles_root = Path(PATH_HOME) / 'data/openrooms_shape_pickles' # for caching shape bboxes so that we do not need to load meshes very time if only bboxes are wanted

    fov = 57.95 / 180.0 * np.pi


    # obj_file = root / 'data/openrooms_raw/uv_mapped/03636649/6512e3a8e6566fd0cf004563556ddb36/aligned_light.obj'
    # obj_file = root / 'test_files/2/5_03636649_6512e3a8e6566fd0cf004563556ddb36_object.obj'
    # print(str(obj_file))

    meta_split = 'main_xml'
    # scene_name = 'scene0008_00_more'; frame_idx = 1; light_name = 'light_3SG-SkyGrd_0-03636649_6512e3a8e6566fd0cf004563556ddb36_object'
    scene_name = 'scene0005_00_more'; frame_idx = 3; light_name = 'light_3SG-SkyGrd_0-ceiling_lamp_15367_object'
    
    # frame_ids = list(range(102))
    frame_ids = [frame_idx]

    root = Path('/Users/jerrypiglet/Documents/Projects/OpenRooms_RAW_loader')
    rendering_dir = root / Path('data/public_re_3/%s/%s'%(meta_split, scene_name))
    light_dir = rendering_dir / ('light_%d'%frame_idx) 

    openrooms_scene = openroomsScene3D(
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'rendering_root': base_root, 'xml_scene_root': xml_root, 'semantic_labels_root': semantic_labels_root, 'shape_pickles_root': shape_pickles_root, 
        'layout_root': layout_root, 'shapes_root': shapes_root, 'envmaps_root': envmaps_root}, 
    scene_params_dict={'meta_split': meta_split, 'scene_name': scene_name, 'frame_id_list': frame_ids}, 
    # modality_list = ['im_sdr', 'im_hdr', 'seg', 'poses', 'albedo', 'roughness', 'depth', 'normal', 'lighting_SG', 'lighting_envmap'], 
    modality_list = [
        'im_sdr', 'poses', 'seg', 
        'im_hdr', 'albedo', 'roughness', 
        'depth', 'normal', 
        # 'lighting_SG', 
        # 'lighting_envmap', 
        'layout', 
        'shapes', # objs + emitters, geometry shapes + emitter properties
        'mi', # mitsuba scene, loading from scene xml file
    ], 
    im_params_dict={
        'im_H_load': 480, 'im_W_load': 640, 
        'im_H_resize': 240, 'im_W_resize': 320
        }, 
    shape_params_dict={
        'if_load_obj_mesh': False, # set to False to not load meshes for objs (furniture) to save time
        'if_load_emitter_mesh': True,  # default True: to load emitter meshes, because not too many emitters
        },
    mi_params_dict={
        'if_also_dump_xml_with_lit_lamps_only': True,  # True: to dump a second file containing lit-up lamps only
        'debug_dump_mesh': True, # [DEBUG] True: to dump all object meshes to mitsuba/meshes_dump; load all .ply files into MeshLab to view the entire scene: images/demo_mitsuba_dump_meshes.png
        'debug_render_test_image': False, # [DEBUG][slow] True: to render an image with first camera, usig Mitsuba: images/demo_mitsuba_render.png
        'if_sample_rays_pts': True, # True: to sample camera rays and intersection pts given input mesh and camera poses
        'if_get_segs': True, # True: to generate segs similar to those in openroomsScene2D.load_seg()
        },
    )
    

    with open(str(light_dir / 'box0.dat'), 'rb') as fIn:
        lightBox = pickle.load(fIn)

    with open(str(rendering_dir / ('%s.dat'%light_name)), 'rb') as fIn:
        lightSrc = pickle.load(fIn)
    
    # find the lamp
    lamp_matches = False
    lamp_matches_id = None
    for lamp_idx, (lamp, _, _) in enumerate(openrooms_scene.lamp_list):
        if np.amax(np.abs(np.array(lightSrc['intensity']).reshape(3,) - np.array(lamp['emitter_prop']['intensity']).reshape(3,))) < 1e-3:
            assert lamp['id'] == lightSrc['shapeId']
            # print(lamp['filename'])
            assert lamp_matches == False
            lamp_matches = True
            lamp_matches_id = lamp_idx
    assert lamp_matches
    _, vertices, faces = openrooms_scene.lamp_list[lamp_matches_id]

    # vertices, faces = loadMesh(obj_file)
    vertices -= openrooms_scene.pose_list[0][:3, 3:4].reshape((1, 3))
    vertices = vertices @ (openrooms_scene.pose_list[0][:3, :3])
    vertices = np.hstack([vertices[:, 0:1], -vertices[:, 1:2], -vertices[:, 2:3]])
    writeMesh('tmp_lamp.obj', vertices, faces) # [!!!] 'tmp_lamp.obj' and 'lamp_0_0.obj' should roughly match in coordinates: demo_lamp_match.png

    center = lightBox['box3D_cam']['center']
    center = center.reshape(1, 3).astype(np.float32)
    center = torch.from_numpy(center)

    intensity = lightSrc['intensity']
    intensity = intensity.reshape(1, 3).astype(np.float32)
    intensity = torch.from_numpy(intensity)

    height, width = 240, 320

    pixel_len = np.tan(fov / 2) / width * 2.0
    pixel_len = pixel_len * np.abs(center.numpy().squeeze()[2])

    normal = cv2.imread(str(rendering_dir / ('imnormal_%d.png'%frame_idx)))[:, :, ::-1]
    normal = np.ascontiguousarray(normal)
    normal = cv2.resize(normal,(width, height), interpolation = cv2.INTER_AREA)

    normal = normal.astype(np.float32)
    normal = normal / 127.5 - 1
    normal = normal.transpose(2, 0, 1)
    normal = normal / np.sqrt(
            np.maximum(np.sum(normal * normal, axis=0, keepdims=True), 1e-12))

    normal = normal.reshape(1, 3, height, width)
    normal = torch.from_numpy(normal)

    mask = cv2.imread(str(light_dir / 'mask0.png'))[:, :, 0]
    mask = cv2.resize(mask, (width, height), interpolation = cv2.INTER_AREA)
    mask = (mask == 255)
    mask = ndimage.binary_erosion(mask, structure = np.ones((3, 3)))

    mask = mask.reshape(1, 1, height, width)
    mask = torch.from_numpy(mask)

    print(str(light_dir / 'mask0.png'))

    with open(str(rendering_dir / ('imdepth_%d.dat'%frame_idx)), 'rb') as fIn:
        hBuffer = fIn.read(4)
        dh = struct.unpack('i', hBuffer)[0]
        wBuffer = fIn.read(4)
        dw = struct.unpack('i', wBuffer)[0]

        dBuffer = fIn.read()
        depth = struct.unpack('f' * dh * dw, dBuffer)
        depth = np.array(depth).reshape(dh, dw).astype(np.float32)

    depth = cv2.resize(depth, (width, height), interpolation = cv2.INTER_AREA)

    depth = depth.reshape(1, 1, height, width)
    depth = torch.from_numpy(depth)

    renderer = renderDirecLighting(isCuda = False)
    shading, lpts_arr = renderer.forward(center, intensity,
            depth, mask, normal, isTest = False, 
            visLampMeshNames=['tmp_lamp.obj'], 
            )

    shading = shading.numpy().reshape(3, height, width)
    shading = shading.transpose(1, 2, 0)[:, :, ::-1]
    cv2.imwrite('shading.hdr', shading)

    # Visualize the geometry; will fail if the lamp is not visible in image
    utils.writeLampList([center], depth, normal, mask, 1, 'lamp.ply')
