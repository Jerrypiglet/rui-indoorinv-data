from abc import abstractproperty, abstractmethod
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

class rendererBase():
    '''
    A class used to visualize/render Mitsuba scene in XML format
    '''
    def __init__(
        self, 
        mitsuba_scene, 
        modality_list: list, 
        im_params_dict: dict={}, 
        cam_params_dict: dict={}, 
        mi_params_dict: dict={}, 
    ):
        self.os = mitsuba_scene
        self.modality_list = self.check_and_sort_modalities(list(set(modality_list)))

        self.scene_rendering_path = self.os.scene_rendering_path
        self.im_params_dict = {**self.os.im_params_dict, **im_params_dict}
        self.cam_params_dict = {**self.os.cam_params_dict, **cam_params_dict}
        self.mi_params_dict = {**self.os.mi_params_dict, **mi_params_dict}

    @abstractproperty
    def valid_modalities(self):
        ...

    @property
    def modality_folder_maping(self):
        return {
            'im': 'Image', 
            'depth': 'Depth',
            'normal': 'Normal',
            # 'Alpha',
            # 'IndexOB',
            'albedo': 'DiffCol',
            # 'GlossCol',
            'emission': 'Emit', 
            'roughness': 'Roughness', 
            # 'Metallic'
            }

    @property
    def modality_filename_maping(self):
        return {
            'im': ['*_0001.exr', '%03d_0001.exr'], 
            'depth': ['', ''], 
            'normal': ['', ''],
            # 'Alpha',
            # 'IndexOB',
            'albedo': ['', ''],
            # 'GlossCol',
            'emission': ['', ''], 
            'roughness': ['', ''], 
            # 'Metallic'
            }

    def check_and_sort_modalities(self, modalitiy_list):
        modalitiy_list_new = [_ for _ in self.valid_modalities if _ in modalitiy_list]
        for _ in modalitiy_list_new:
            assert _ in self.valid_modalities, 'Invalid modality: %s'%_
        return modalitiy_list_new

    @abstractmethod
    def render(self):
        ...

    def render_modality_check(self, modality, force=False):
        assert modality in self.modality_folder_maping
        folder_name = self.modality_folder_maping[modality]
        render_folder_path = self.scene_rendering_path / folder_name
        assert modality in self.modality_filename_maping
        filename_pattern = self.modality_filename_maping[modality][0]

        if_render = 'y'
        files = sorted(glob.glob(str(render_folder_path / filename_pattern)))
        if force:
            if_render = 'y'
        else:
            if len(files) > 0:
                if_render = input(red("[%s] %d %s files found at %s. Re-render? [y/n]"%(modality, len(files), filename_pattern, str(render_folder_path))))
        if if_render in ['N', 'n']:
            print(yellow('ABORTED rendering by Mitsuba'))
            return
        else:
            if render_folder_path.exists():
                shutil.rmtree(str(render_folder_path))
            render_folder_path.mkdir(parents=True, exist_ok=True)
            print(yellow('Files removed from %s'%str(render_folder_path)))

        return folder_name, render_folder_path

