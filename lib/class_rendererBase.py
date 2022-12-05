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

    def check_and_sort_modalities(self, modalitiy_list):
        modalitiy_list_new = [_ for _ in self.valid_modalities if _ in modalitiy_list]
        for _ in modalitiy_list_new:
            assert _ in self.valid_modalities, 'Invalid modality: %s'%_
        return modalitiy_list_new

    @abstractmethod
    def render(self):
        ...