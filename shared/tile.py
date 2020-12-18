from __future__ import annotations #https://stackoverflow.com/questions/33837918/type-hints-solve-circular-dependency

import sys
sys.path.append('../')
sys.path.append('../tile_extraction/')

from typing import List, Callable, Tuple, Dict

import pathlib
from pathlib import Path
Path.ls = lambda x: [p for p in list(x.iterdir()) if '.ipynb_checkpoints' not in p.name]

from enum import Enum
import tile_extraction
from tile_extraction import tiles
from tiles import *
import numpy
import numpy as np

import PIL
import os

class Tile:
    """
    Class with information about a tile.
    """
    __removed = False #Flag that can be set True, to mark it as "deleted" for the patient_manager. use getter and setter method
                      # this flag is not used in the TileSummary class
    
    tile_summary = None
    tiles_folder_path = None
    np_scaled_filtered_tile = None
    tile_num = None
    r = None # (=row) e.g. the wsi has a height of 1024 pixel and one tile has a height of 256 pixels, r can be in range [1,4] (ends 
             # included); if rois are specified, it's the row number inside the roi; so tiles from different rois can have the same
             # r and c value but with respect to different rois
    c = None # (=column) like r but according to width
    
    r_s = None #(=row_start)pixel value on y-axis of the SCALED down wsi; always with respect to the wsi nevertheless rois are 
                # specified
    r_e = None # (=row_end)
    c_s = None # (=column_start) like r_s but x-axis
    c_e = None #(=column_end)
    
    o_r_s = None #(=original_row_start)pixel value on y-axis of the UNscaled wsi on the specified level; always with respect to the 
                    # wsi nevertheless rois are specified
    o_r_e = None #(=original_row_end)
    o_c_s = None #(=original_column_start)
    o_c_e = None #(=original_column_end)
    
    tissue_percentage = None #tissue percentage
    color_factor = None
    s_and_v_factor = None
    quantity_factor = None
    score = None
    tile_naming_func = None
    level = None
    best_level_for_downsample = None
    real_scale_factor = None
    roi:RegionOfInterest = None
    tile_path = None
    labels:List[Union[str,int]] = None # y true
    labels_one_hot_encoded:numpy.ndarray = None
    predictions_raw:Dict[str,float] = None # key: class name; value: predicted probability
    predictions_thresh:Dict[str, bool] = None # key: class name; value: bool
    loss:float = None
                
    def __init__(self, 
                 tile_summary=None, 
                 tiles_folder_path=None, 
                 np_scaled_filtered_tile=None, 
                 tile_num=None, 
                 r=None, 
                 c=None, 
                 r_s=None, 
                 r_e=None, 
                 c_s=None, 
                 c_e=None, 
                 o_r_s=None, 
                 o_r_e=None, 
                 o_c_s=None,
                 o_c_e=None, 
                 t_p=None, #tissue_percentage
                 color_factor=None, 
                 s_and_v_factor=None, 
                 quantity_factor=None, 
                 score=None, 
                 tile_naming_func=None, 
                 level=None,
                 best_level_for_downsample=None,
                 real_scale_factor=None,
                 roi:RegionOfInterest=None,
                 tile_path = None, 
                 labels:List[int] = None):
        """
        Arguments:
            level: whole-slide image's level, the tile shall be extracted from
            best_level_for_downsample: openslide.OpenSlide.get_best_level_for_downsample(scale_factor)
            best_level_for_downsample: openslide.OpenSlide.get_best_level_for_downsample(scale_factor)
        """


        self.tile_summary = tile_summary
        self.roi = roi
        self.tiles_folder_path = tiles_folder_path
        self.np_scaled_filtered_tile = np_scaled_filtered_tile
        self.tile_num = tile_num
        self.r = r
        self.c = c
        self.r_s = r_s
        self.r_e = r_e
        self.c_s = c_s
        self.c_e = c_e
        self.o_r_s = o_r_s
        self.o_r_e = o_r_e
        self.o_c_s = o_c_s
        self.o_c_e = o_c_e
        self.tissue_percentage = t_p
        self.color_factor = color_factor
        self.s_and_v_factor = s_and_v_factor
        self.quantity_factor = quantity_factor
        self.score = score
        self.tile_naming_func = tile_naming_func
        self.level = level
        self.best_level_for_downsample = best_level_for_downsample
        self.real_scale_factor = real_scale_factor
        self.tile_path = tile_path
        self.labels = labels

    def __str__(self):
        if(self.tile_path != None):
            return str(self.tile_path.name)
        else:
            wsi_name = None
            try:
                wsi_name = self.tile_summary.wsi_path.name
            except:
                wsi_name = 'to be set'
                
            return f'wsi: {wsi_name}; '+"[Tile #%d, Row #%d, Column #%d, Tissue %4.2f%%, Score %0.4f]" % (
              self.tile_num, self.r, self.c, self.tissue_percentage, self.score)

    def __repr__(self):
        return "\n" + self.__str__()

    def is_removed(self):
        return self.__removed
    
    def set_removed_flag(self, value:bool):
        self.__removed = value
            
    def mask_percentage(self):
        return 100 - self.tissue_percentage

    def tissue_quantity(self):
        return tiles.tissue_quantity(self.tissue_percentage)

    def get_pil_tile(self):
        return tiles.tile_to_pil_tile(self)

    def get_np_tile(self):
        return tiles.tile_to_np_tile(self)

    def save_tile(self):
        tiles.save_display_tile(self, save=True, display=False)

    def display_tile(self):
        tiles.save_display_tile(self, save=False, display=True)

    def display_with_histograms(self):
        tiles.display_tile(self, rgb_histograms=True, hsv_histograms=True)

    def get_np_scaled_filtered_tile(self):
        return self.np_scaled_filtered_tile

    def get_pil_scaled_filtered_tile(self):
        return util.np_to_pil(self.np_scaled_filtered_tile)
    
    def get_width(self):
        return self.o_c_e - self.o_c_s
    
    def get_height(self):
        return self.o_r_e - self.o_r_s
    
    def get_x(self):
        """
        upper left x coordinate
        """
        return self.o_c_s
    
    def get_y(self):
        """
        upper left y coordinate
        """
        return self.o_r_s
    
    def get_path(self)->pathlib.Path:
        return pathlib.Path(tiles.get_tile_image_path(self))
                  
    def get_name(self)->str:
        return pathlib.Path(tiles.get_tile_image_path(self)).name
    
    def get_dataset_type(self)->shared.enums.DatasetType:
        return self.roi.whole_slide_image.case.patient.dataset_type
    
    def get_wsi_path(self)->pathlib.Path:
        return self.tile_summary.wsi_path
    
    def calculate_predictions_ohe(self, thresholds:Dict[str, float])->numpy.ndarray:
        """
        Calculates predictions based on the given thresholds after the predictions_raw had been set.
        Arguments:
        
        Returns:
            1-d numpy array representing one hot encoded predictions.
        """
        assert self.predictions_raw != None
        assert list(self.predictions_raw.keys()) == list(thresholds.keys())
        return (np.array(list(self.predictions_raw.values())) >= np.array(list(thresholds.values()))).astype(np.int0)
    
    def get_labels(self)->List[Union[str,int]]:
        return self.labels
    
    def get_predictions_one_hot_encoded(self)->numpy.ndarray:
        """
            Returns:
                numpy array with one hot encoded labels
        """
        return np.array(list(self.predictions_thresh.values())).astype(np.int0)
    
    def get_labels_one_hot_encoded(self)->numpy.ndarray:
        """
            Returns:
                numpy array with one hot encoded labels
        """
        return np.array(self.labels_one_hot_encoded).astype(np.int0)
        
    def get_pil_image(self)->PIL.PngImagePlugin.PngImageFile:
        # tile is saved to disc => just open it
        if(self.tile_path != None and os.path.exists(self.tile_path)):
            return PIL.Image.open(self.tile_path)
        # tile needs to be extracted from its corresponding wsi (or preextracted roi)
        else:
            return tiles.ExtractTileFromWSI(path=self.get_wsi_path(), 
                                             x=self.get_x(), 
                                             y=self.get_y(), 
                                             width=self.get_width(), 
                                             height=self.get_height(), 
                                             level=self.level)