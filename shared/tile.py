from __future__ import annotations #https://stackoverflow.com/questions/33837918/type-hints-solve-circular-dependency

import pathlib
from pathlib import Path
Path.ls = lambda x: [p for p in list(x.iterdir()) if '.ipynb_checkpoints' not in p.name]

from enum import Enum
from tile_extraction import tiles
from tiles import *



class Tile:
    """
    Class for information about a tile.
    """
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
    
    t_p = None #tissue percentage
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
    labels = None
                
    def __init__(self, 
                 tile_summary, 
                 tiles_folder_path, 
                 np_scaled_filtered_tile, 
                 tile_num, 
                 r, 
                 c, 
                 r_s, 
                 r_e, 
                 c_s, 
                 c_e, 
                 o_r_s, 
                 o_r_e, 
                 o_c_s,
                 o_c_e, 
                 t_p, 
                 color_factor, 
                 s_and_v_factor, 
                 quantity_factor, 
                 score, 
                 tile_naming_func, 
                 level,
                 best_level_for_downsample,
                 real_scale_factor,
                 roi:RegionOfInterest,
                 tile_path = None):
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

    def __str__(self):
        return "[Tile #%d, Row #%d, Column #%d, Tissue %4.2f%%, Score %0.4f]" % (
          self.tile_num, self.r, self.c, self.tissue_percentage, self.score)

    def __repr__(self):
        return "\n" + self.__str__()

    def mask_percentage(self):
        return 100 - self.tissue_percentage

    def tissue_quantity(self):
        return tissue_quantity(self.tissue_percentage)

    def get_pil_tile(self):
        return tile_to_pil_tile(self)

    def get_np_tile(self):
        return tile_to_np_tile(self)

    def save_tile(self):
        save_display_tile(self, save=True, display=False)

    def display_tile(self):
        save_display_tile(self, save=False, display=True)

    def display_with_histograms(self):
        display_tile(self, rgb_histograms=True, hsv_histograms=True)

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
        return pathlib.Path(get_tile_image_path(self))
                  
    def get_name(self)->str:
        return pathlib.Path(get_tile_image_path(self)).name