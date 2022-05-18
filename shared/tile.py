from __future__ import annotations #https://stackoverflow.com/questions/33837918/type-hints-solve-circular-dependency

import sys
sys.path.append('../')
sys.path.append('../tile_extraction/')

from typing import List, Callable, Tuple, Dict

import pathlib
from pathlib import Path
Path.ls = lambda x: [p for p in list(x.iterdir()) if '.ipynb_checkpoints' not in p.name]

from enum import Enum
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
    
    tilesummary = None
    tiles_folder_path = None
    grid_manager = None
    #contains spatial information about the tile with respect to the user specified WSI level
    rectangle:Rectangle = None
    rectangle_downsampled:Rectangle = None
    score = None
    dict_with_all_parameters_to_determine_score = None
    tile_naming_func = None
    level = None
    level_downsampled = None
    real_scale_factor = None
    roi:RegionOfInterest = None
    tile_path = None
    labels:List[Union[str,int]] = None # y true
    labels_one_hot_encoded:numpy.ndarray = None
    predictions_raw:Dict[str,float] = None # key: class name; value: predicted probability
    predictions_thresh:Dict[str, bool] = None # key: class name; value: bool
    
    #result from learner.predict from exported learner
    predictions_fastai_inference = None # e.g.: ((#1) ['Ganglioglioma'], tensor([False,  True]), tensor([0.0013, 0.9987]))
    loss:float = None
    
                
    def __init__(self, 
                 tilesummary, 
                 tiles_folder_path, 
                 tile_num,
                 grid_manager:GridManager,
                 rectangle:Rectangle,
                 rectangle_downsampled:Rectangle,
                 score,
                 dict_with_all_parameters_to_determine_score,
                 tile_naming_func, 
                 level,
                 level_downsampled,
                 real_scale_factor,
                 roi:RegionOfInterest):
        """
        Arguments:
            level: whole-slide image's level, the tile shall be extracted from
            level_downsampled: openslide.OpenSlide.get_best_level_for_downsample(scale_factor)
            dict_with_all_parameters_to_determine_score: A dictionary which contains all factors 
                                                         that were relevant for determining the score
                                                         inside of the tile_scoring_function. This dict is 
                                                         returned by the tile_scoring_function.
        """


        self.tilesummary = tilesummary
        self.roi = roi
        self.tiles_folder_path = tiles_folder_path
        self.tile_num = tile_num
        self.grid_manager = grid_manager
        self.rectangle=rectangle
        self.rectangle_downsampled=rectangle_downsampled
        self.score = score
        self.dict_with_all_parameters_to_determine_score = dict_with_all_parameters_to_determine_score
        self.tile_naming_func = tile_naming_func
        self.level = level
        self.level_downsampled = level_downsampled
        self.real_scale_factor = real_scale_factor

    def __str__(self):
        if(self.tile_path != None):
            return str(self.tile_path.name)
        else:
            wsi_name = None
            try:
                wsi_name = self.tilesummary.wsi_path.name
            except:
                wsi_name = 'to be set'
                
            return f'wsi: {wsi_name}; '+"[Tile #%d, Score %0.4f]" % (self.tile_num, self.score)

    def __repr__(self):
        return "\n" + self.__str__()

    def is_removed(self):
        return self.__removed
    
    def set_removed_flag(self, value:bool):
        self.__removed = value        

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
    
    def get_width(self):
        return self.rectangle.width()
    
    def get_height(self):
        return self.rectangle.height()
    
    def get_x(self):
        """
        upper left x coordinate
        """
        return self.rectangle.ul.x
    
    def get_y(self):
        """
        upper left y coordinate
        """
        return self.rectangle.ul.y
    
    def get_path(self)->pathlib.Path:
        return pathlib.Path(tiles.get_tile_image_path(self))
                  
    def get_name(self)->str:
        #return pathlib.Path(tiles.get_tile_image_path(self)).name
        return self.__str__()
    
    def get_dataset_type(self)->shared.enums.DatasetType:
        return self.roi.whole_slide_image.case.patient.dataset_type
    
    def get_wsi_path(self)->pathlib.Path:
        return self.tilesummary.wsi_path
    
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
            return PIL.Image.open(self.get_path())
        # tile needs to be extracted from its corresponding wsi (or preextracted roi)
        else:
            wh = WsiHandler(self.get_wsi_path())
            return wh.extract_tile_from_wsi_2(rectangle_tile=self.rectangle, 
                                              level=self.level)
        
        
###################################### more imports ################################################################
import tile_extraction
from tile_extraction import tiles
from tiles import *