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
        if_skip_check: bool=False,
        im_params_dict: dict={}, 
        cam_params_dict: dict={}, 
        mi_params_dict: dict={}, 
        lighting_params_dict: dict={}, 
    ):
        self.os = mitsuba_scene
        self.modality_list = self.check_and_sort_modalities(list(set(modality_list)))
        
        self.if_skip_check = if_skip_check

        self.scene_rendering_path = self.os.scene_rendering_path
        self.im_params_dict = {**self.os.CONF.im_params_dict, **im_params_dict}
        self.cam_params_dict = {**self.os.CONF.cam_params_dict, **cam_params_dict}
        self.mi_params_dict = {**self.os.CONF.mi_params_dict, **mi_params_dict}
        self.lighting_params_dict = {**self.os.CONF.lighting_params_dict, **lighting_params_dict}

    @abstractproperty
    def valid_modalities(self):
        ...

    @property
    def modality_folder_maping(self):
        return {
            'im': 'Image', 
            'depth': 'Depth',
            'normal': 'Normal',
            'geo_normal': 'GeoNormal',
            'sh_normal': 'ShadingNormal',
            # 'Alpha',
            'albedo': 'DiffCol',
            'specular_color': 'GlossCol',
            
            'emission': 'Emit', 
            'roughness': 'Roughness', 
            'lighting_envmap': 'LightingEnvmap', 
            'metallic': 'Metallic', 
            'invalid_mat': 'InvalidMat',
            
            'mat_index': 'IndexMA', 
            'shape_index': 'IndexOB',
            'uv': 'UV', 
            'position': 'Position', 
            }

    @property
    def modality_filename_maping(self):
        '''
        for Blender rendering; or Mitsuba rendering, remember to manually append: '_mi'
        '''
        return {
            'im': ['*_0001.exr', '%03d_0001.exr'], 
            'lighting_envmap': ['*_0001.exr', '%03d_0001.exr'], 
            'depth': ['*_0001.exr', '%03d_0001.exr'], 
            'normal': ['*_0001.exr', '%03d_0001.exr'], 
            'geo_normal': ['*_0001.exr', '%03d_0001.exr'], 
            'sh_normal': ['*_0001.exr', '%03d_0001.exr'], 
            'uv': ['*_0001.exr', '%03d_0001.exr'], 
            'position': ['*_0001.exr', '%03d_0001.exr'], 
            'shape_index': ['*_0001.exr', '%03d_0001.exr'], 
            # 'Alpha',
            'mat_index': ['*_0001.exr', '%03d_0001.exr'], 
            
            'albedo': ['*_0001.exr', '%03d_0001.exr'], 
            'specular_color': ['*_0001.exr', '%03d_0001.exr'], 
            
            'emission': ['*_0001.exr', '%03d_0001.exr'], 
            'roughness': ['*_0001.exr', '%03d_0001.exr'], 
            'metallic': ['*_0001.exr', '%03d_0001.exr'], 
            'invalid_mat': ['*_0001.exr', '%03d_0001.exr'], 
            }

    def check_and_sort_modalities(self, modalitiy_list):
        modalitiy_list_new = [_ for _ in self.valid_modalities if _ in modalitiy_list]
        for _ in modalitiy_list_new:
            assert _ in self.valid_modalities, 'Invalid modality: %s'%_
        return modalitiy_list_new

    @abstractmethod
    def render(self):
        ...

    def render_modality_check(self, modality, folder_name_appendix='', if_force=False, file_name_appendix=''):
        assert modality in self.modality_folder_maping, 'Invalid modality not found in modality_folder_maping: %s'%modality
        folder_name = self.modality_folder_maping[modality] + folder_name_appendix
        render_folder_path = self.scene_rendering_path / folder_name
        assert modality in self.modality_filename_maping, 'Invalid modality not found in modality_filename_maping: %s'%modality
        filename_pattern = self.modality_filename_maping[modality][0].split('.')[0] + file_name_appendix + '.' + self.modality_filename_maping[modality][0].split('.')[1]

        if_render = 'y'
        files = sorted(glob.glob(str(render_folder_path / filename_pattern)))
        if not self.if_skip_check:
            if if_force:
                if_render = 'y'
            else:
                if len(files) > 0:
                    if_render = input(red("[%s] %d %s files found at %s. RE-RENDER? [y/n]"%(modality, len(files), filename_pattern, str(render_folder_path))))
        if if_render in ['N', 'n']:
            print(yellow('ABORTED rendering by Mitsuba'))
            return
        else:
            if render_folder_path.exists():
                if if_force:
                    if_remove = 'Y'
                elif self.if_skip_check:
                    if_remove = 'N'
                else:
                    if_remove = input(red("[%s] %d %s files found at %s. REMOVE? [y/n]"%(modality, len(files), filename_pattern, str(render_folder_path))))
                if if_remove in ['Y', 'y']:
                    # shutil.rmtree(str(render_folder_path), ignore_errors=True)
                    for file in files:
                        Path(file).unlink()
                        file_png = Path(file).parent / (Path(file).name.replace('.exr', '.png'))
                        if file_png.exists():
                            file_png.unlink()
                    print(yellow('Files removed from %s'%str(render_folder_path)))
            render_folder_path.mkdir(parents=True, exist_ok=True)

        return folder_name, render_folder_path

    def compare_blender_mi_Image(self):
        modality = 'Image'
        folder_name = self.modality_folder_maping[modality]
        render_folder_path = self.scene_rendering_path / folder_name
        assert render_folder_path.exists(), 'Folder not found: %s'%str(render_folder_path)
        filename_pattern = self.modality_filename_maping[modality][0]

        files = sorted(glob.glob(str(render_folder_path / filename_pattern)))
        for file in files:
            file_mi = file.parent / (file.name + '_mi' + file.suffix)
            import ipdb; ipdb.set_trace()

