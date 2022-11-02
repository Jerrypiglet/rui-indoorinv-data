from webbrowser import BackgroundBrowser
import numpy as np
import time
from lib.class_openroomsScene2D import openroomsScene2D
from lib.class_openroomsScene3D import openroomsScene3D
import matplotlib.pyplot as plt
from lib.utils_vis import vis_index_map, colorize
from lib.utils_OR.utils_OR_lighting import converter_SG_to_envmap
from lib.utils_OR.utils_OR_cam import project_3d_line
from lib.utils_OR.utils_OR_lighting import downsample_lighting_envmap
class visualizer_openroomsScene_2D(object):
    '''
    A class used to **visualize** **per-pixel** OpenRooms (public/public-re versions) scene contents (2D/2.5D per-pixel DENSE properties / semantics).
    '''
    def __init__(
        self, 
        openrooms_scene, 
        modality_list_vis: list, 
        frame_idx_list: list=[0], 
    ):

        assert type(openrooms_scene) in [openroomsScene2D, openroomsScene3D], '[visualizer_openroomsScene] has to take an object of openroomsScene or openroomsScene3D!'

        self.os = openrooms_scene

        self.modality_list_vis = list(set(modality_list_vis))
        for _ in self.modality_list_vis:
            assert _ in self.valid_modalities_2D_vis, 'Invalid modality: %s'%_

        self.frame_idx_list = frame_idx_list
        self.N_frames = len(self.frame_idx_list)
        assert self.N_frames >= 1

        self.N_cols = self.N_frames
        assert self.N_cols <= 6 # max 6 images due to space in a row
        # self.N_rows = len(self.modality_list_vis) + 1

        self.semseg_colors = np.loadtxt('data/colors/openrooms_colors.txt').astype('uint8')
        if any([_ in ['lighting_SG'] for _ in self.modality_list_vis]):
            self.converter_SG_to_envmap = converter_SG_to_envmap(
                SG_num=self.os.lighting_params_dict['SG_num'], 
                env_width=self.os.lighting_params_dict['env_width'], 
                env_height=self.os.lighting_params_dict['env_height']
                )

    @property
    def valid_modalities_2D_vis(self):
        return [
            'im', 
            'albedo', 'roughness', 'depth', 'normal', 
            'lighting_SG', # convert to lighting_envmap and vis
            'lighting_envmap', 
            'semseg', 'matseg', 'seg_area', 'seg_env', 'seg_obj', 
            'mi_depth', 'mi_normal', 'mi_seg_area', 'mi_seg_env', 'mi_seg_obj', 
            'layout', 'shapes', 
            ]

    def create_im_row_ax_list(self, subfig, start_idx: int=1, if_show_im: bool=False, title: str=''):
        assert self.os.if_has_im_sdr

        ax_list = subfig.subplots(1, self.N_cols)
        if self.N_cols == 1: ax_list = [ax_list]
        assert len(self.frame_idx_list) == len(ax_list)
        for ax, frame_idx in zip(ax_list, self.frame_idx_list):
            if if_show_im:
                im = self.os.im_sdr_list[frame_idx]
                ax.imshow(im)

        start_idx += len(self.frame_idx_list)

        return ax_list, start_idx

    def vis_2d_with_plt(
        self, 
        **kwargs, 
        ):
        '''
        visualize verything indicated in modality_list for the frame_idx-st frame (0-based)
        '''
        height_width_list = []
        assert self.os.if_has_im_sdr
        for frame_idx in self.frame_idx_list:
            im = self.os.im_sdr_list[frame_idx]
            height, width = im.shape[:2]
            height_width_list.append((height, width))

        compatible_modalities = ['im'] + self.valid_modalities_2D_vis
        # modality_list_show = [_ for _ in compatible_modalities if _ in ['im']+self.modality_list_vis]
        modality_list_show = [_ for _ in self.modality_list_vis if _ in compatible_modalities]
        if 'im' not in modality_list_show: modality_list_show = ['im']+modality_list_show

        start_idx = 1
        # plt.figure(figsize=(6*self.N_cols, 4*self.N_rows))
        fig = plt.figure(constrained_layout=True)
        subfigs = fig.subfigures(nrows=len(modality_list_show), ncols=1) # https://stackoverflow.com/questions/27426668/row-titles-for-matplotlib-subplot

        for subfig in subfigs:
            modality = modality_list_show.pop(0)
            modality_title_appendix = ''

            if modality == 'im':
                _, start_idx = self.create_im_row_ax_list(subfig, start_idx, if_show_im=True)
            else:
                if_show_im = False
                if modality == 'layout':
                    if_show_im = True

                ax_list, start_idx = self.create_im_row_ax_list(subfig, start_idx, if_show_im=if_show_im)

                if modality == 'layout':
                    self.vis_2d_layout(ax_list)
                    for frame_idx, ax, (height, width) in zip(self.frame_idx_list, ax_list, height_width_list):
                        ax.set_xlim(-width*0.5, width*1.5)
                        ax.set_ylim(height*1.5, -height*.5)
                else:
                    # other modalities
                    self.vis_2d_modality(fig=subfig, ax_list=ax_list, modality=modality, **kwargs)

                    if modality == 'albedo':
                        modality_title_appendix = '(in SDR space)'
                    if modality == 'matseg':
                        modality_title_appendix = '(red for invalid areas (e.g. emitters)'
                    if modality in ['mi_depth', 'mi_normal', 'mi_seg']:
                        modality_title_appendix = '(from Mitsuba)'

            subfig.suptitle(modality+' '+modality_title_appendix)

        plt.show()

    def vis_2d_modality(
        self, 
        fig, 
        ax_list, 
        modality, 
        lighting_params={
            'lighting_scale': 0.1, 
            }, 
        ):
        '''
        visualize 2D map for the modality the frame_idx-st frame (0-based)

        '''
        assert self.os.if_has_im_sdr and self.os.if_has_poses
        if modality in ['depth', 'normal']: assert self.os.if_has_dense_geo
        if modality in ['albedo', 'roughness']: assert self.os.if_has_BRDF
        if modality in ['seg_area', 'seg_env', 'seg_obj']: assert self.os.if_has_seg
        if modality in ['mi_depth', 'mi_normal', 'mi_seg_area', 'mi_seg_env', 'mi_seg_obj']: assert self.os.if_has_mitsuba_scene

        _list = self.os.get_modality(modality)
        for frame_idx, ax in zip(self.frame_idx_list, ax_list):
            _im = _list[frame_idx]

            if modality == 'normal':
               _im = (_im + 1.) / 2. 
            if modality == 'mi_normal':
                assert self.os.pts_from['mi']
                R = self.os.pose_list[frame_idx][:3, :3]
                mi_normal_global = _im
                mi_normal_cam = (R.T @ mi_normal_global.reshape(-1, 3).T).T.reshape(self.os.H, self.os.W, 3)
                # transform mi_normal from OpenCV (right-down-forward) to OpenGL convention (right-up-backward)
                mi_normal_cam = np.stack([mi_normal_cam[:, :, 0], -mi_normal_cam[:, :, 1], -mi_normal_cam[:, :, 2]], axis=-1)
                _im = np.clip((mi_normal_cam+1.)/2., 0., 1.)
                _im[mi_normal_global==np.inf] = 0.

            if modality == 'depth':
                plot = ax.imshow(_im, vmin=0., cmap='jet')
                plt.colorbar(plot, ax=ax)
                continue
            if modality == 'mi_depth':
                assert self.os.pts_from['mi']
                vmin, vmax = np.amin(self.os.depth_list[frame_idx]), np.amax(self.os.depth_list[frame_idx])
                _im[_im==np.inf] = 0.
                plot = ax.imshow(_im, vmin=0., vmax=vmax, cmap='jet')
                plt.colorbar(plot, ax=ax)
                continue

            if modality == 'albedo':
                # convert albedo to SDR for better vis
               _im = _im ** (1./2.2) 

            if modality == 'matseg':
                _im = vis_index_map(_im['mat_aggre_map'])
            if modality == 'semseg':
                _im = np.array(colorize(_im, self.semseg_colors).convert('RGB'))

            if modality == 'lighting_SG':
                axis_local, lamb, weight = np.split(_im, [3, 4], axis=3)
                if self.os.if_has_HDR_scale:
                    weight = weight / self.os.hdr_scale_list[frame_idx]
                # ts = time.time()
                envmap_cam = self.converter_SG_to_envmap.convert_converter_SG_to_envmap_2D(axis_local, lamb, weight) # -> (120, 160, 3, 8, 16)
                # print('----', time.time() - ts)
                # ts = time.time()
                # envmap_cam = self.converter_SG_to_envmap.convert_converter_SG_to_envmap_2D_np(axis_local, lamb, weight) # -> (120, 160, 3, 8, 16)
                # print('====', time.time() - ts)
                lighting_scale = lighting_params.get('lighting_scale', 0.1)
                _im = np.clip(downsample_lighting_envmap(envmap_cam, lighting_scale=lighting_scale)**(1./2.2), 0., 1.)

            if modality == 'lighting_envmap':
                if self.os.if_has_HDR_scale:
                    _im = _im / self.os.hdr_scale_list[frame_idx]
                lighting_scale = lighting_params.get('lighting_scale', 0.1)
                _im = np.clip(downsample_lighting_envmap(_im, lighting_scale=lighting_scale)**(1./2.2), 0., 1.)

            ax.imshow(_im)

    def vis_2d_layout(self, ax_list):
        '''
        visualize projected layout for the frame_idx-st frame (0-based)

        images/demo_layout_3D_proj.png

        '''
        assert self.os.if_has_im_sdr and self.os.if_has_poses
        assert self.os.if_has_layout

        for frame_idx, ax in zip(self.frame_idx_list, ax_list):
            R_c2w, t_c2w = self.os.pose_list[frame_idx][:3, :3], self.os.pose_list[frame_idx][:3, 3:4]
            R_w2c = np.linalg.inv(R_c2w)
            t_w2c = - R_w2c @ t_c2w
            (origin, zaxis, _) = self.os.origin_lookatvector_up_list[frame_idx]
            
            for idx_list in [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
            # for idx_list in [[6, 7]]:
                v3d_array = self.os.layout_box_3d_transformed

                for i in range(len(idx_list)-1):
                    # print(idx_list[i], idx_list[i+1])
                    x1x2 = np.vstack((v3d_array[idx_list[i]], v3d_array[idx_list[i+1]]))
                    x1x2_proj = project_3d_line(x1x2, R_w2c, t_w2c, self.os.K, origin, zaxis, dist_cam_plane=0., if_debug=False)
                    if x1x2_proj is not None:
                        ax.plot([x1x2_proj[0][0], x1x2_proj[1][0]], [x1x2_proj[0][1], x1x2_proj[1][1]], color='r', linewidth=1)   
                        ax.text(x1x2_proj[0][0], x1x2_proj[0][1], str(idx_list[i]), backgroundcolor='w')
                        ax.text(x1x2_proj[1][0], x1x2_proj[1][1], str(idx_list[i+1]), backgroundcolor='w')
                        # ax.text((x1x2_proj[0][0]+x1x2_proj[1][0])/2., (x1x2_proj[0][1]+x1x2_proj[1][1])/2., str(idx_list[i])+'-'+str(idx_list[i+1]), backgroundcolor='r')

