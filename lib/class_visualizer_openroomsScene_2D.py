from webbrowser import BackgroundBrowser
import numpy as np
import open3d as o3d
from lib.utils_misc import blue_text, get_list_of_keys, green, white_blue, red, check_list_of_tensors_size
from lib.class_openroomsScene import openroomsScene
from lib.class_openroomsScene3D import openroomsScene3D
import matplotlib.pyplot as plt
from lib.utils_OR.utils_OR_cam import project_3d_line

class visualizer_openroomsScene_2D(object):
    '''
    A class used to **visualize** OpenRooms (public/public-re versions) scene contents (2D/2.5D per-pixel DENSE properties / semantics).
    '''
    def __init__(
        self, 
        openrooms_scene, 
        modality_list: list, 
        frame_idx_list: list=[0], 
    ):

        assert type(openrooms_scene) in [openroomsScene, openroomsScene3D], '[visualizer_openroomsScene] has to take an object of openroomsScene or openroomsScene3D!'

        self.openrooms_scene = openrooms_scene

        self.modality_list = modality_list
        for _ in self.modality_list:
            assert _ in ['layout']

        self.frame_idx_list = frame_idx_list
        self.N_frames = len(self.frame_idx_list)
        assert self.N_frames >= 1

        self.N_cols = 4
        self.N_rows = self.N_frames // self.N_cols + 1 # max 4 images / row

    def create_im_ax_list(self):
        assert self.openrooms_scene.if_has_im_sdr

        plt.figure(figsize=(6*self.N_cols, 4*self.N_rows))
        ax_list = []
        for frame_idx in self.frame_idx_list:
            ax = plt.subplot(self.N_rows, self.N_cols, frame_idx+1)
            im = self.openrooms_scene.im_sdr_list[frame_idx]
            ax.imshow(im)
            ax_list.append(ax)

        return ax_list

    def vis_2d_with_plt(self, ax_list=None):
        '''
        visualize verything indicated in modality_list for the frame_idx-st frame (0-based)
        '''
        if ax_list is None:
            ax_list = self.create_im_ax_list()

        if 'layout' in self.modality_list:
            self.vis_2d_layout(ax_list)

        for frame_idx, ax in zip(self.frame_idx_list, ax_list):
            im = self.openrooms_scene.im_sdr_list[frame_idx]
            height, width = im.shape[:2]
            ax.set_xlim(-width*0.5, width*1.5)
            ax.set_ylim(height*1.5, -height*.5)

        plt.show()


    def vis_2d_layout(self, ax_list=None):
        '''
        visualize projected layout for the frame_idx-st frame (0-based)

        images/demo_layout_3D_proj.png

        '''
        assert self.openrooms_scene.if_has_im_sdr and self.openrooms_scene.if_has_cameras

        if ax_list is None:
            ax_list = self.create_im_ax_list()

        for frame_idx, ax in zip(self.frame_idx_list, ax_list):
            R_c2w, t_c2w = self.openrooms_scene.pose_list[frame_idx][:3, :3], self.openrooms_scene.pose_list[frame_idx][:3, 3:4]
            R_w2c = np.linalg.inv(R_c2w)
            t_w2c = - R_w2c @ t_c2w
            (origin, zaxis, _) = self.openrooms_scene.origin_lookat_up_list[frame_idx]
            
            for idx_list in [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
            # for idx_list in [[6, 7]]:
                v3d_array = self.openrooms_scene.layout_box_3d_transformed

                for i in range(len(idx_list)-1):
                    # print(idx_list[i], idx_list[i+1])
                    x1x2 = np.vstack((v3d_array[idx_list[i]], v3d_array[idx_list[i+1]]))
                    x1x2_proj = project_3d_line(x1x2, R_w2c, t_w2c, self.openrooms_scene.K, origin, zaxis, dist_cam_plane=0., if_debug=False)
                    if x1x2_proj is not None:
                        ax.plot([x1x2_proj[0][0], x1x2_proj[1][0]], [x1x2_proj[0][1], x1x2_proj[1][1]], color='r', linewidth=1)   
                        ax.text(x1x2_proj[0][0], x1x2_proj[0][1], str(idx_list[i]), backgroundcolor='w')
                        ax.text(x1x2_proj[1][0], x1x2_proj[1][1], str(idx_list[i+1]), backgroundcolor='w')
                        # ax.text((x1x2_proj[0][0]+x1x2_proj[1][0])/2., (x1x2_proj[0][1]+x1x2_proj[1][1])/2., str(idx_list[i])+'-'+str(idx_list[i+1]), backgroundcolor='r')

