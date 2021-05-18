# ------------------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------

# To get around renderer issue on macOS going from Matplotlib image to NumPy image.
import matplotlib

matplotlib.use('Agg')

import PIL
import pathlib
from pathlib import Path
import colorsys
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy
import numpy as np
import os
import PIL
from PIL import Image, ImageDraw, ImageFont
from enum import Enum
import openslide
from typing import List, Callable, Union, Dict, Tuple, Union
from tqdm import tqdm
import pandas
import pandas as pd
import warnings
from enum import Enum
import shapely
import copy
from functools import partial
import pathos

import util, filter, slide, openslide_overwrite
from util import *
import shared
from shared import roi
from shared.tile import Tile
from shared.roi import *
from shared.enums import DatasetType, TissueQuantity




TISSUE_HIGH_THRESH = 80
TISSUE_LOW_THRESH = 10
HSV_PURPLE = 270
HSV_PINK = 330

############################# classes #########################################


class TileSummary:
    """
    Class for tile summary information.
    """

    wsi_path = None
    tiles_folder_path = None #only necessary, if the tiles shall be saved to disc, else None   
    orig_w = None #full width in pixels of the wsi on the specified level
    orig_h = None #full height in pixels of the wsi on the specified level
    orig_tile_w = None 
    orig_tile_h = None
    
    minimal_acceptable_tile_width = None #between 0.0 and 1.0; percentage of orig_w ;tiles at the edges will not have full width
    minimal_acceptable_tile_height = None #between 0.0 and 1.0; percentage of orig_h ;tiles at the edges will not have full height
    
    scale_factor = None #for faster processing the wsi is scaled down internally, the resulting tiles are on maximum resolution
                                #depending on the specified level
    scaled_w = None 
    scaled_h = None
    scaled_tile_w = None
    scaled_tile_h = None
    mask_percentage = None
    num_row_tiles = None
    num_col_tiles = None
    tile_score_thresh = None
    level = None
    best_level_for_downsample = None
    real_scale_factor = None
        
    rois:RegionOfInterest
    
    tiles = None

    count = 0
    high = 0
    medium = 0
    low = 0
    none = 0

    def __init__(self, 
                 wsi_path,
                 tiles_folder_path, 
                 orig_w, 
                 orig_h, 
                 orig_tile_w, 
                 orig_tile_h,
                 minimal_acceptable_tile_width,
                 minimal_acceptable_tile_height,
                 scale_factor, 
                 scaled_w, 
                 scaled_h, 
                 scaled_tile_w,
                 scaled_tile_h, 
                 tissue_percentage, 
                 num_col_tiles, 
                 num_row_tiles, 
                 tile_score_thresh, 
                 level, 
                 best_level_for_downsample,
                 real_scale_factor, 
                 rois:List[RegionOfInterest]):

        """
        Arguments:
            level: whole-slide image's level, the tiles shall be extracted from
            orig_w, orig_h: original height and original width depend on the specified level. With each level, the dimensions half.
            
            minimal_acceptable_tile_width: factor between 0.0 and 1.0. percentage of orig_w ;affects tiles at the edges, which cannot 
                                            have full width
            minimal_acceptable_tile_height: factor between 0.0 and 1.0. percentage of orig_h ;affects tiles at the edges, which 
                                            cannot have full height                                                               
                                                                       
            scale_factor:  Downscaling is applied during tile calculation to speed up the process. The tiles in the end get extracted 
                            from the full resolution. the full resolution depends on the level, the user specifies. The higher the 
                            level, the lower the resolution/magnification.
                                 Therefore less downsampling needs to be applied during tile calculation, to achieve same speed up.
                                 So e.g. the wsi has dimensions of 10000x10000 pixels on level 0. A scale_factor of 32 is speficied. 
                                 Then calculations will be applied on
                                 a downscaled version of the wsi with dimensions on the level log2(32)

            real_scale_factor: if a scale_factor of e.g. 32 is specified and a level of 0, from which the tiles shall be extracted, 
                                scale_factor==real_scale_factor.
                                 For each level, the wsi dimensions half.
                                 That means for a scale_factor of 32 and level 1 the real_scale_factor would be only 16.
                                 downscaling is applied during tile calculation to speed up the process. The tile in the end get 
                                 extracted from the full resolution
                                 The full resolution depends on the level, the user specifies. The higher the level, the lower the 
                                 resolution/magnification.
                                 Therefore less downsampling needs to be applied during tile calculation, to achieve same speed up.

            best_level_for_downsample: result of openslide.OpenSlide.get_best_level_for_downsample(scale_factor)
        """

        self.wsi_path = wsi_path
        self.tiles_folder_path = tiles_folder_path
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.orig_tile_w = orig_tile_w
        self.orig_tile_h = orig_tile_h       
        self.minimal_acceptable_tile_width = minimal_acceptable_tile_width
        self.minimal_acceptable_tile_height = minimal_acceptable_tile_height        
        self.scale_factor = scale_factor
        self.scaled_w = scaled_w
        self.scaled_h = scaled_h
        self.scaled_tile_w = scaled_tile_w
        self.scaled_tile_h = scaled_tile_h
        self.tissue_percentage = tissue_percentage
        self.num_col_tiles = num_col_tiles
        self.num_row_tiles = num_row_tiles
        self.tile_score_thresh = tile_score_thresh
        self.level = level
        self.best_level_for_downsample = best_level_for_downsample
        self.real_scale_factor = real_scale_factor
        
        self.tiles = []
        
        self.rois = rois

        
    def change_level_of_rois(self, new_level:int):
        """
            convenience function to change the level for all rois at once in place
        """
        for roi in self.rois:
            if roi != None:
                roi.change_level_in_place(new_level)

    #def __str__(self):
    #    return summary_title(self) + "\n" + summary_stats(self)

    def mask_percentage(self):
        """
        Obtain the percentage of the slide that is masked.

        Returns:
           The amount of the slide that is masked as a percentage.
        """
        return 100 - self.tissue_percentage

    def num_tiles(self):
        """
        Retrieve the total number of tiles.

        Returns:
          The total number of tiles (number of rows * number of columns).
        """
        return self.num_row_tiles * self.num_col_tiles

    def tiles_by_tissue_percentage(self):
        """
        Retrieve the tiles ranked by tissue percentage.

        Returns:
           List of the tiles ranked by tissue percentage.
        """
        sorted_list = sorted(self.tiles, key=lambda t: t.tissue_percentage, reverse=True)
        return sorted_list

    def tiles_by_score(self):
        """
        Retrieve the tiles ranked by score. If rois were specified, only tiles within those rois will be taken into account.

        Returns:
           List of the tiles ranked by score.
        """
        sorted_list = sorted(self.tiles, key=lambda t: t.score, reverse=True)
        return sorted_list

    def get_tile(self, row, col):
        """
        Retrieve tile by row and column.

        Args:
          row: The row
          col: The column

        Returns:
          Corresponding Tile object.
        """
        tile_index = (row - 1) * self.num_col_tiles + (col - 1)
        tile = self.tiles[tile_index]
        return tile
    
    def top_tiles(self, verbose=False):
        """
        Retrieve only the tiles that pass scoring and their height and width are greater than 
        minimal_acceptable_tile_height/minimal_acceptable_tile_width.

        Returns:
           List of the top-scoring tiles.
        """
        sorted_tiles = self.tiles_by_score()
        top_tiles = [tile for tile in sorted_tiles
                     if self.check_tile(tile)]
        if verbose:
            print(f'{self.wsi_path}: Number of tiles that will be kept/all possible tiles: {len(top_tiles)}/{len(sorted_tiles)}')
        return top_tiles

    def check_tile(self, tile):
        width = tile.o_c_e - tile.o_c_s
        height = tile.o_r_e - tile.o_r_s
        return tile.score > self.tile_score_thresh and width >= self.minimal_acceptable_tile_width*self.orig_tile_w and height >= \
                self.minimal_acceptable_tile_height*self.orig_tile_h
        
    
    def show_wsi(self, 
                 figsize:tuple=(10,10), 
                 axis_off:bool=False):
        """
        Displays a scaled down overview image of the wsi.
        
        Arguments:
            figsize: Size of the plotted matplotlib figure containing the image.
            axis_off: bool value that indicates, if axis shall be plotted with the picture
        """
        wsi_pil, large_w, large_h, new_w, new_h, best_level_for_downsample = wsi_to_scaled_pil_image(wsi_filepath=self.wsi_path, 
                                                                                                               scale_factor=self.scale_factor, 
                                                                                                               level=0)                                                               
        # Create figure and axes
        fig,ax = plt.subplots(1,1,figsize=figsize)    
        # Display the image
        ax.imshow(wsi_pil)     
        if(axis_off):
            ax.axis('off') 
        plt.show() 
    
    
    def show_wsi_with_top_tiles(self, 
                                   figsize:Tuple[int] = (10,10),
                                   scale_factor:int = 32, 
                                   axis_off:bool = False):
        """    
        Loads a whole slide image, scales it down, converts it into a numpy array and displays it with a grid overlay for all tiles
        that passed scoring to visualize which tiles e.g. "tiles.WsiOrROIToTilesMultithreaded" calculated as worthy to keep.
        Arguments:
            figsize: Size of the plotted matplotlib figure containing the image.
            scale_factor: The larger, the faster this method works, but the plotted image has less resolution.
            axis_off: bool value that indicates, if axis shall be plotted with the picture
        """
        wsi_pil, large_w, large_h, new_w, new_h, best_level_for_downsample = wsi_to_scaled_pil_image(self.wsi_path,                                                                                                scale_factor=self.scale_factor,
                                                                                                     level=0)                                                                  
        wsi_np = util.pil_to_np_rgb(wsi_pil)
        
        tiles_np = []
        for tile in self.top_tiles():
            x = util.adjust_level(tile.get_x(), tile.level, best_level_for_downsample)
            y = util.adjust_level(tile.get_y(), tile.level, best_level_for_downsample)
            width = util.adjust_level(tile.get_width(), tile.level, best_level_for_downsample)
            height = util.adjust_level(tile.get_height(), tile.level, best_level_for_downsample)          
            tiles_np.append(np.array([[x,y],[x+width, y],[x+width, y+height],[x, y+height]]))
            
        rois_np = []
        for roi in self.rois:
            roi_adjusted = roi.change_level_deep_copy(new_level = best_level_for_downsample)
            rois_np.append(util.polygon_to_numpy(roi_adjusted.polygon))
                
        util.show_np_img_with_polygons(img=wsi_np, polygons_np=tiles_np+rois_np, figsize=figsize, axis_off=axis_off)

                                 
    def show_wsi_with_rois(self, 
                           figsize:Tuple[int] = (10,10),
                           scale_factor:int = 32, 
                           axis_off:bool = False):
        """    
        Loads a whole slide image, scales it down, converts it into a numpy array and displays it with a grid overlay for all rois
        specified in self.wsi_info.rois
        Arguments:
            figsize: Size of the plotted matplotlib figure containing the image.
            scale_factor: The larger, the faster this method works, but the plotted image has less resolution.
            axis_off: bool value that indicates, if axis shall be plotted with the picture
        """
        util.show_wsi_with_rois(self.wsi_path, self.rois, axis_off=axis_off)





class Vertex:
    def __init__(self, x:float, y:float):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f'(x:{self.x}, y:{self.y})'
    
    def __str__(self):
        return f'(x:{self.x}, y:{self.y})'
    
    def __add__(self, o):
        if(type(o) is np.ndarray and (o.shape == (2,) or o.shape == (2,1))):
            self.x += o[0]
            self.y += o[1]
        else:
            raise TypeError(f'Vertex class does not support addition with type: {type(o)}')

    def __sub__(self, o):
        if(type(o) is np.ndarray and (o.shape == (2,) or o.shape == (2,1))):
            self.x -= o[0]
            self.y -= o[1]
        else:
            raise TypeError(f'Vertex class does not support subtraction with type: {type(o)}')
           
    def __mul__(self, o):
        if(type(o) is int or type(o) is float):
            self.x *= o
            self.y *= o
        else:
            raise TypeError(f'Vertex class does not support multiplication with type: {type(o)}')
            
    def __rmatmul__(self, o):
        if(type(o) is np.ndarray):
            return o@np.array([self.x, self.y])
        else:
            raise TypeError(f'Vertex class does not support (right sided) matrix multiplication with type: {type(o)}')
    
    def __call__(self):
        return np.array([self.x, self.y])
        
    def rotate_around_pivot(self, angle:float, pivot = np.array([0, 0])):
        """
        Rotates itself clockwise around the specified pivot with the specified angle.
        """
        self.__add__(-pivot)
        radians = math.radians(angle)
        rotation_matrix = np.array([[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]])
        new_coordinates = rotation_matrix@self.__call__()
        self.x = new_coordinates[0]
        self.y = new_coordinates[1]
        self.__add__(pivot)

        
class Rectangle:
    def __init__(self, 
                 upper_left:Vertex, 
                 upper_right:Vertex, 
                 lower_right:Vertex, 
                 lower_left:Vertex):
        self.ul = upper_left
        self.ur = upper_right
        self.lr = lower_right
        self.ll = lower_left
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return f'(ul: {self.ul}, ur: {self.ur}, lr: {self.lr}, ll: {self.ll})'
    
    def __call__(self):
        return np.array([self.ul(), self.ur(), self.lr(), self.ll()])
    
    def __add_to_all_vertices(self, o:np.ndarray):
        self.ul + o
        self.ur + o
        self.lr + o
        self.ll + o
    
    def __rotate_all_vertices(self, angle:float, pivot:np.ndarray):
        self.ul.rotate_around_pivot(angle=angle, pivot=pivot)
        self.ur.rotate_around_pivot(angle=angle, pivot=pivot)
        self.lr.rotate_around_pivot(angle=angle, pivot=pivot)
        self.ll.rotate_around_pivot(angle=angle, pivot=pivot)
    
    def deepcopy(self):
        ul_dc = copy.deepcopy(self.ul)
        ur_dc = copy.deepcopy(self.ur)
        lr_dc = copy.deepcopy(self.lr)
        ll_dc = copy.deepcopy(self.ll)
        return Rectangle(ul_dc,ur_dc,lr_dc,ll_dc)
    
    def polygon(self):
        return shapely.geometry.Polygon(np.array([self.ul(), self.ur(), self.lr(), self.ll()]))
    
    def rotate_around_itself(self, angle:float):
        """
        Rotates itself around its centroid.
        Arguments:
            angle: degrees clockwise
        """
        centroid = np.array([self.polygon().centroid.x, self.polygon().centroid.y])
        #rotate all four vertices around this new pivot
        self.__rotate_all_vertices(angle=angle, pivot=centroid)
        
    def rotate_around_pivot_without_orientation_change(self, angle:float, pivot = np.array([0,0])):
        """
        Rotates the rectangle's centroid around the specified pivot.
        The orientations of the edges do not change. 
        Arguments:
            angle: degrees clockwise
        """
        self.__add_to_all_vertices(-pivot)
        
        centroid_old = Vertex(x=self.polygon().centroid.x, y=self.polygon().centroid.y)
        centroid_new = copy.deepcopy(centroid_old)
        centroid_new.rotate_around_pivot(angle=angle, pivot=np.array([0,0]))
        self.__add_to_all_vertices(-centroid_old())
        self.__add_to_all_vertices(centroid_new())
        
        self.__add_to_all_vertices(pivot)
    
    def rotate(self, angle:float, pivot = np.array([0,0])):
        """
        Combines rotate_around_itself and rotate_around_pivot
        Arguments:
            angle: degrees clockwise
        """
        self.rotate_around_itself(angle=angle)
        self.rotate_around_pivot_without_orientation_change(angle=angle, pivot=pivot)
    
    def as_roi(self, level = 0)->shared.roi.RegionOfInterestPolygon:
        """
        Creates and returns a RegionOfInterestPolygon from its values.
        Arguments:
            level: WSI level
        """
        return shared.roi.RegionOfInterestPolygon(roi_id=self.__str__(), 
                                                  vertices=self.__call__(), 
                                                  level = level)        
class Grid:
    def __init__(self, 
                 min_width:int, 
                 min_height:int, 
                 tile_width:int, 
                 tile_height:int,
                 coordinate_origin_x:int = 0, 
                 coordinate_origin_y:int = 0, 
                 angle:int = 0):
        ##
        #init object attributes
        ##
        self.min_width = min_width
        self.min_height = min_height
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.coordinate_origin_x = coordinate_origin_x
        self.coordinate_origin_y = coordinate_origin_y
        
        self.grid_centroid_float = np.array([coordinate_origin_x+min_width/2, coordinate_origin_y+min_height/2])
        self.grid_centroid_int = np.array([int(coordinate_origin_x+min_width/2), int(coordinate_origin_y+min_height/2)])
        self.init_angle = angle
        self.current_angle = 0 #init with 0 and a value != 0 will be set in the rotate method 
                                #at the end of the constructor 
        
        ##
        #calculate grid of rectangles
        ##        
        self.__init_grid(angle=self.init_angle)
    
    def __init_grid(self, angle):
        n_rows = math.ceil(self.min_height/self.tile_height)+1
        n_columns = math.ceil(self.min_width/self.tile_width)+1
        
        # the grid needs to be larger, to still cover the image after rotation
        wsi_diagonal = math.sqrt(self.min_width**2 + self.min_height**2)
        n_extra_rows = self.__round_up_to_nearest_even((wsi_diagonal-self.min_height)/self.tile_height)
        n_extra_columns = self.__round_up_to_nearest_even((wsi_diagonal-self.min_width)/self.tile_width)
                
        # init grid with None
        self.grid = np.zeros(shape=(n_rows+n_extra_rows, n_columns+n_extra_columns), dtype=Rectangle)
        
        for c in range(-int(n_extra_columns/2), n_columns + int(n_extra_columns/2)):
            for r in range(-int(n_extra_rows/2), n_rows + int(n_extra_rows/2)):
                #upper left vertex
                ul_x = self.coordinate_origin_x + self.tile_width*c
                ul_y = self.coordinate_origin_y + self.tile_height*r
                ul = Vertex(x=ul_x, y=ul_y)
                #upper right vertex
                ur_x = self.coordinate_origin_x + self.tile_width*(c+1)
                ur_y = self.coordinate_origin_y + self.tile_width*r
                ur = Vertex(x=ur_x, y=ur_y)
                #lower right vertex
                lr_x = self.coordinate_origin_x + self.tile_width*(c+1)
                lr_y = self.coordinate_origin_y + self.tile_width*(r+1)
                lr = Vertex(x=lr_x, y=lr_y)
                #lower left vertex
                ll_x = self.coordinate_origin_x + self.tile_width*c
                ll_y = self.coordinate_origin_y + self.tile_width*(r+1)
                ll = Vertex(x=ll_x, y=ll_y)
                
                self.grid[r + int(n_extra_rows/2)][c + int(n_extra_columns/2)] = Rectangle(upper_left=ul, 
                                                                                            upper_right=ur, 
                                                                                            lower_right=lr, 
                                                                                            lower_left=ll)       
        if(angle%360 != 0):
            self.rotate(angle=angle)
    
    
    def __round_up_to_nearest_even(self, n:float)->int:
        n = math.ceil(n)
        if(n%2 == 0):
            return n
        else:
            return n+1
    
    def reset_grid(self):
        self.current_angle = 0
        self.__init_grid(angle=self.init_angle)
        
    def get_rectangles(self)->List[Rectangle]:
        return [rect for rect in self.grid.flatten() if type(rect) is Rectangle]
    
    def get_number_of_tiles(self)->int:
        def __predicate(elem):
            if(elem is None):
                return False
            return True
        return np.count_nonzero(np.where(__predicate, self.grid, 1)) 
    
    def as_rois(self)->List[shared.roi.RegionOfInterestPolygon]:
        def __func(rect):
            if(type(rect) is Rectangle):
                return rect.as_roi()
        return [__func(rect) for rect in self.grid.flatten() if type(rect) is Rectangle]
    
    def rotate(self, angle:float):
        """
        Rotates the grid around the center of the wsi.
        Arguments:
            angle: angle in degrees, clockwise
        """
        if(angle%360 != 0):
            def __func(rect):
                if(type(rect) is Rectangle):
                    rect.rotate(angle=angle, pivot=self.grid_centroid_float)
    
            f = np.vectorize(__func)
            f(self.grid)
            #update current angle
            self.current_angle = (self.current_angle+angle)%360
        
    def filter_grid(self, 
                   roi:shared.roi.RegionOfInterestPolygon, 
                    minimal_intersection_quotient:float ):
        """
        Arguments:
            roi: a region of interest inside the WSI; it will be checked, if the tiles lay inside of it.
            minimal_intersection_quotient: in the range of (0.0, 1.0], only tiles with a relative 
                                           intersection with the roi equal or above this threshold 
                                           will be kept
        """
        if(minimal_intersection_quotient <= 0.0 or minimal_intersection_quotient > 1.0):
            raise ValueError("minimal_intersection_quotient must be in range (0.0, 1.0]")
            
        for row in range(self.grid.shape[0]):
            for col in range (self.grid.shape[1]):
                elem = self.grid[row][col]
                if(type(elem) is Rectangle):
                    rect_as_roi = elem.as_roi()
                    intersection_area = roi.polygon.intersection(rect_as_roi.polygon).area
                    tile_area = rect_as_roi.polygon.area
                    intersection_quotient = intersection_area/tile_area
                    #remove tile from grid, if the intersection with the roi is too small
                    if(intersection_quotient < minimal_intersection_quotient):
                        self.grid[row][col] = None
        
class GridManager:
    def __init__(self, 
                 wsi_path:pathlib.Path, 
                 tile_width:int, 
                 tile_height:int,
                 rois:List[shared.roi.RegionOfInterestPolygon], 
                 grids_per_roi:int = 1):
        """
        Arguments:
            wsi_path:
            tile_width:
            tile_height:
            grids_per_roi: Use multiple grids per roi to enhance the number of tiles.
                            The grids will be shifted depending on the number of grids 
                            and the tile_width, tile_height.
                            e.g.: grids_per_roi == 3 and tile_width == tile_height == 1024
                                  First grid starts at (0,0).
                                  Second grid starts at (1024*1/3 , 1024*1/3)
                                  Third grid starts at (1024*2/3 , 1024*2/3)
            rois: If no roi is specified, the complete WSI is implicitly considered as one roi.
        """
        self.wsi_path = wsi_path
        self.tile_width = tile_width
        self.tile_height = tile_height
        if(len(rois) > 1):
            rois_dc = [copy.deepcopy(r) for r in rois]
            shared.roi.merge_overlapping_rois(rois=rois_dc)
            self.rois = rois_dc
        else:
            self.rois = rois
        self.grids_per_roi = grids_per_roi
        self.grids = []
        self.roi_to_grids = {}
        self.grid_to_roi = {}
        if(self.rois is None or len(self.rois) == 0):
            w = slide.open_slide(path=wsi_path)
            ul = np.array([0,0])
            ur = np.array([wsi.dimensions[0], 0])
            lr = np.array([wsi.dimensions[0], wsi.dimensions[1]])
            ll = np.array([0, wsi.dimensions[1]])
            vertices = np.array([ul, ur, lr, ll])
            r = RegionOfInterestPolygon(roi_id=f'{wsi_path.stem} - dummy_roi', vertices=vertices)
            for i in range(grids_per_roi):
                shift_origin_x = tile_width*i/grids_per_roi
                shift_origin_y = tile_height*i/grids_per_roi
                g = Grid(min_width=wsi.dimensions[0], 
                         min_height=wsi.dimensions[1], 
                         tile_width=tile_width, 
                         tile_height=tile_height, 
                         coordinate_origin_x=shift_origin_x, 
                         coordinate_origin_y=shift_origin_y)
                self.__add_grid(g=g, r=r)
        else:
            for r in self.rois:
                b_ul = Vertex(x=r.polygon.bounds[0], y=r.polygon.bounds[1])
                b_ur = Vertex(x=r.polygon.bounds[2], y=r.polygon.bounds[1])
                b_lr = Vertex(x=r.polygon.bounds[2], y=r.polygon.bounds[3])
                b_ll = Vertex(x=r.polygon.bounds[0], y=r.polygon.bounds[3])
                b_width = r.polygon.bounds[2] - r.polygon.bounds[0]
                b_height = r.polygon.bounds[3] - r.polygon.bounds[1]
                for i in range(grids_per_roi):
                    shift_origin_x = tile_width*i/grids_per_roi
                    shift_origin_y = tile_height*i/grids_per_roi
                    g = Grid(min_width=b_width, 
                             min_height=b_height, 
                             tile_width=tile_width, 
                             tile_height=tile_height, 
                             angle=0, 
                             coordinate_origin_x= r.polygon.bounds[0]+shift_origin_x, 
                             coordinate_origin_y= r.polygon.bounds[1]+shift_origin_y)
                    self.__add_grid(g=g, r=r)
    
    def __add_grid(self, g:Grid, r:RegionOfInterestPolygon):
        self.grids.append(g)
        if(r not in self.roi_to_grids.keys()):
            self.roi_to_grids[r] = []
        self.roi_to_grids[r].append(g)
        self.grid_to_roi[g] = r
    
    
    def show_wsi_with_rois_and_grids(self, 
                                     figsize: Tuple[int] = (10, 10), 
                                     scale_factor: int = 32, 
                                     axis_off: bool = False):
        
        for i in range(self.grids_per_roi):
            tiles_as_rois = []
            for r in self.roi_to_grids.keys():
                tiles_as_rois += self.roi_to_grids[r][i].as_rois()
            util.show_wsi_with_rois(wsi_path=self.wsi_path, 
                                rois=self.rois + tiles_as_rois, 
                                figsize=figsize, 
                                scale_factor=scale_factor, 
                                axis_off=axis_off)    
        
    def filter_grids(self, minimal_intersection_quotient:float):
        """
        Filters out all tiles, which have not a minimum relative intersection 
        of minimal_intersection_quotient with a roi.
        """
        for g in self.grids:
            g.filter_grid(roi=self.grid_to_roi[g], 
                          minimal_intersection_quotient=minimal_intersection_quotient)
            
    def reset_grids(self):
        for g in self.grids:
            g.reset_grid()
        
    def __iteration(self, g:Grid, stepsize:float, minimal_intersection_quotient:float):
            best_angle = 0
            max_num_tls = 0            
            current_angle = 0
            while(current_angle <= 90):
                g.reset_grid()
                g.rotate(angle=current_angle)
                g.filter_grid(roi=self.grid_to_roi[g], 
                            minimal_intersection_quotient=minimal_intersection_quotient)
                current_number_of_tiles = g.get_number_of_tiles()
                if(current_number_of_tiles > max_num_tls):
                    max_num_tls = current_number_of_tiles
                    best_angle = current_angle
                current_angle += stepsize    
                
            #to not forget a rotation of exactly 90°   
            if(g.tile_width != g.tile_height and current_angle > 90):
                g.reset_grid()
                g.rotate(angle=90)
                g.filter_grid(roi=self.grid_to_roi[g], 
                            minimal_intersection_quotient=minimal_intersection_quotient)
                current_number_of_tiles = g.get_number_of_tiles()
                if(current_number_of_tiles > max_num_tls):
                    max_num_tls = current_number_of_tiles
                    best_angle = current_angle
                current_angle += stepsize
                
            g.reset_grid()
            g.rotate(best_angle)
            g.filter_grid(roi=self.grid_to_roi[g], 
                          minimal_intersection_quotient=minimal_intersection_quotient)
        
    def optimize_grid_angles(self, 
                             stepsize:float = 5, 
                             minimal_intersection_quotient:float=1, 
                             num_workers:int = pathos.util.os.cpu_count()):
        """
        Trys different angles from 0° - 90° with intervals of the size "stepsize" to find the best angle, to fit
        in the most tiles.
        Arguments:
            stepsize: stepsizes of angles that are evaluated
            minimal_intersection_quotient: in the range of (0.0, 1.0], only tiles with a relative 
                                           intersection with the roi equal or above this threshold 
                                           will be kept 
        """
        if(stepsize <= 0.0 or stepsize > 90.0):
            raise ValueError('stepsize must be in range (0, 90]')
        if(minimal_intersection_quotient <= 0.0 or minimal_intersection_quotient > 1.0):
            raise ValueError('minimal_intersection_quotient must be in range (0, 1]')
        
        __foo = partial(self.__iteration, 
                        stepsize=stepsize, 
                        minimal_intersection_quotient=minimal_intersection_quotient)
        
        pool = pathos.pools.ThreadPool(num_workers)
        pool.map(__foo, self.grids)
        
      

############################# functions #########################################

###
# some example implementations for functions that other methods below take as arguments
###
def scoring_function_1(tissue_percent, combined_factor):
    """
    use this, if you want tissue with lots of cells (lots of hematoxylin stained tissue)
    """
    return tissue_percent * combined_factor / 1000.0

def scoring_function_2(tissue_percent, combined_factor):
    """
    use this, if you mostly care that there is any tissue in the tile
    """
    return (tissue_percent ** 2) * np.log(1 + combined_factor) / 1000.0

def get_roi_name_from_path_pituitary_adenoma_entities(roi_path):
    path = Path(roi_path)
    split = path.stem.split('-')
    if split[2] == 'HE':
        return f'{split[0]}-{split[1]}-{split[2]}-{split[3]}-{split[4]}'
    else:
        return f'{split[0]}-{split[1]}-{split[2]}-{split[3]}-{split[4]}-{split[5]}'

def get_wsi_name_from_path_pituitary_adenoma_entities(wsi_path):
    path = Path(wsi_path)
    split = path.stem.split('-')
    return f'{split[0]}-{split[1]}-{split[2]}-{split[3]}'

def tile_naming_function_default(wsi_or_roi_path:Union[pathlib.Path, str])-> str:
    """
    Used as default in WsiOrROIToTiles and WsiOrROIToTilesMultithreaded
    See their docu for more info
    """
    p = Path(wsi_or_roi_path)
    return p.stem

###
# end
###


def ExtractTileFromWSI(path:Union[str, pathlib.Path], x:int, y:int, width:int, height:int, level:int)-> PIL.Image:
    """
    Args:
        path: path to wsi
        x: x-coordinate of the upper left pixel. The method assumes, that you know the dimensions of your specified level.
        y: y-coordinate of the upper left pixel. The method assumes, that you know the dimensions of your specified level.
        width: tile width
        height: tile height
        level: Level of the WSI you want to extract the tile from. 0 means highest resolution.
        
    Return:
        tile as PIL.Image as RGB
    """
    s = slide.open_slide(str(path))
    tile_region = s.read_region((x, y), level, (width, height))
    # RGBA to RGB
    pil_img = tile_region.convert("RGB")
    return pil_img

def ExtractTileFromPILImage(path:Union[str, pathlib.Path], x:int, y:int, width:int, height:int)-> PIL.Image:
    """
    Args:
        path: path to PIL Image
        x: x-coordinate of the upper left pixel
        y: y-coordinate of the upper left pixel
        width: tile width
        height: tile height
        
    Return:
        tile as PIL.Image as RGB
    """
    #pil_img = PIL.Image.open(path)
    #pil_img = pil_img.crop((x, y, x+width, y+height))
    #return pil_img
    return ExtractTileFromWSI(path=path, x=x, y=y, width=width, height=height, level=0);


def WsiOrROIToTiles(wsi_path:pathlib.Path,
                    shifted_grid:bool,
               tiles_folder_path:pathlib.Path,
               tile_height:int, 
               tile_width:int,
               minimal_acceptable_tile_height:float = 0.7,
               minimal_acceptable_tile_width:float = 0.7,
               tile_naming_func:Callable = tile_naming_function_default,
               tile_score_thresh:float = 0.55,
               tile_scoring_function = scoring_function_1,
               level = 0, 
               save_tiles:bool = False, 
               return_as_tilesummary_object = True, 
               rois:List[shared.roi.RegionOfInterestPolygon] = None,
               minimal_tile_roi_intersection_ratio:float = 1.0,     
               verbose=False)-> Union[TileSummary, pandas.DataFrame]:
    raise DeprecationWarning('The function WsiOrRoiToTiles was renamed to WsiToTiles')


def WsiToTiles(wsi_path:pathlib.Path,
               shifted_grid:bool,
               tile_height:int, 
               tile_width:int,
               minimal_acceptable_tile_height:float = 0.7,
               minimal_acceptable_tile_width:float = 0.7,
               tile_naming_func:Callable = tile_naming_function_default,
               tile_score_thresh:float = 0.55,
               tile_scoring_function = scoring_function_1,
               level = 0, 
               save_tiles:bool = False,
               tiles_folder_path:pathlib.Path = None,
               return_as_tilesummary_object = True, 
               rois:List[shared.roi.RegionOfInterestPolygon] = None,
               minimal_tile_roi_intersection_ratio:float = 1.0,     
               verbose=False)-> Union[TileSummary, pandas.DataFrame]:
    """
    Calculates tile coordinates and returns a TileSummary object. If save_tiles == True the tiles will also be extracted
    and saved from the WSI or ROI (ROI is assumed to be a "normal" image format like .png).
    
    Arguments:
    wsi_path: Path to a WSI or ROI(=already extracted part of a wsi in e.g. .png format)
    shifted_grid: During the process at first all possible tile locations are calculated starting at (0,0) (upper left corner)
                    of the wsi. If shifted grid is True, there will be 6 more grids starting at (1/3*tile_width, 0),
                    (2/3*tile_width, 0), (0, 1/3*tile_height), (0, 2/3*tile_height), (1/3*tile_width, 1/3*tile_height),
                    (2/3*tile_width, 2/3*tile_height)
    tile_heigth: Number of pixels tile height.
    tile_width: Number of pixels tile width.
    minimal_acceptable_tile_height: factor between 0.0 and 1.0. percentage of orig_h ;affects tiles at the edges, which 
                                    cannot have full height 
    minimal_acceptable_tile_width: factor between 0.0 and 1.0. percentage of orig_w ;affects tiles at the edges, which cannot 
                                    have full width
    
    tile_score_thresh: Tiles with a score higher than the number from "tileScoringFunction" will be saved.
    tile_scoring_function: Function to score one tile to determine if it should be saved or not.
    tile_naming_func: 99% of the time there should be no necessity to change this.
                        A function, that takes a pathlib.Path to the WSI or ROI as an argument and returns a string.
                        This string will then be used as part of the name for the tile (plus some specific tile information and
                        the file format .png, whick is generated by this library).
                        
    level: Level of the WSI you want to extract the tiles from. 0 means highest resolution. For not wsi formats like .png leave it at 
            0.
    save_tiles: if True the tiles will be extracted and saved to {tilesFolderPath}
    tiles_folder_path: The folder where the extracted tiles will be saved (only needed if save_tiles=True).
    return_as_tilesummary_object: return_as_tilesummary_object: Set this to true, if you 
                                    want the TileSummary object and not a pandas dataframe.
    rois: If rois are specified, only tissue inside these rois will be extracted
    minimal_tile_roi_intersection_ratio: [0.0, 1.0] 
                                         (intersection area between roi and tile)/tile area >= tile_roi_intersection_ratio so 
                                          that a tile will be used for further calculations
    Return:
    if return_as_tilesummary_object == True:
       a TileSummary object will be returned
    else:
        pandas dataframe with coloumns: ['tile_name','wsi_path','level','x_upper_left','y_upper_left','pixels_width','pixels_height']
    """    

    if(tiles_folder_path is None and save_tiles == True):
        raise ValueError("You should specify a {tiles_folder_path}")
    
    if verbose:
        print(f"Starting to process {str(wsi_path)}")

    scale_factor = 32

    ### against DecompressionBombWarning
    #mage.MAX_IMAGE_PIXELS = 10000000000000
    openslide.lowlevel._load_image = openslide_overwrite._load_image

    img_pil, original_width, original_height, scaled_width, scaled_height, best_level_for_downsample = wsi_to_scaled_pil_image(wsi_path, scale_factor, level)

                
    img_pil_filtered = filter.filter_img(img_pil)
    tilesummary = create_tilesummary(wsiPath=wsi_path,
                                     shifted_grid=shifted_grid,
                                     tiles_folder_path=tiles_folder_path, 
                                     img_pil=img_pil, 
                                     img_pil_filtered=img_pil_filtered, 
                                     wsi_original_width=original_width, 
                                     wsi_original_height=original_height, 
                                     wsi_scaled_width=scaled_width, 
                                     wsi_scaled_height=scaled_height, 
                                     tile_height=tile_height, 
                                     tile_width=tile_width,
                                     minimal_acceptable_tile_height=minimal_acceptable_tile_height,
                                     minimal_acceptable_tile_width=minimal_acceptable_tile_width,
                                     scale_factor=scale_factor,
                                     tile_score_thresh=tile_score_thresh,
                                     tile_scoring_function=tile_scoring_function,
                                     tile_naming_func=tile_naming_func, 
                                     level=level, 
                                     best_level_for_downsample=best_level_for_downsample, 
                                     rois=rois,
                                     minimal_tile_roi_intersection_ratio=minimal_tile_roi_intersection_ratio,
                                     verbose=verbose)
    
    if(save_tiles):
        for tile in tilesummary.top_tiles(verbose):
            tile.save_tile()
            
    if return_as_tilesummary_object:
        if verbose:
            tilesummary.top_tiles(verbose)
        return tilesummary
    
    else:    
        rows_list = []
        for tile in tilesummary.top_tiles(verbose):                                      
            row = {'tile_name':tile.get_name(),
                'wsi_path':tile.tile_summary.wsi_path,
                'level':tile.level,
                'x_upper_left':tile.get_x(),
                'y_upper_left':tile.get_y(),
                'pixels_width':tile.get_width(),
                'pixels_height':tile.get_height()}
            rows_list.append(row)
        
        if(len(rows_list) == 0):
            return pd.DataFrame(columns=['tile_name','wsi_path', \
                                         'level','x_upper_left','y_upper_left','pixels_width','pixels_height'])
        else:
            return pd.DataFrame(rows_list).set_index('tile_name', inplace=False)
        
def WsiOrROIToTilesMultithreaded(wsi_paths:List[pathlib.Path],
                             shifted_grid:bool,
                             tiles_folder_path:pathlib.Path,
                             tile_height:int, 
                             tile_width:int,
                             minimal_acceptable_tile_height:float = 0.7,
                             minimal_acceptable_tile_width:float = 0.7,
                             tile_naming_func:Callable = tile_naming_function_default,
                             tile_score_thresh:float = 0.55,
                             tile_scoring_function = scoring_function_1,  
                             level = 0, 
                             save_tiles:bool = False, 
                             return_as_tilesummary_object = False, 
                             wsi_path_to_rois:Dict[pathlib.Path, List[shared.roi.RegionOfInterestPolygon]] = None,
                             minimal_tile_roi_intersection_ratio:float = 1.0,
                             verbose=False)-> Union[List[TileSummary], pandas.DataFrame]:
    raise DeprecationWarning('The function WsiOrROIToTilesMultithreaded was renamed to WsisToTilesParallel')
        
        
      
def WsisToTilesParallel(wsi_paths:List[pathlib.Path],
                             shifted_grid:bool,
                             tile_height:int, 
                             tile_width:int,
                             minimal_acceptable_tile_height:float = 0.7,
                             minimal_acceptable_tile_width:float = 0.7,
                             tile_naming_func:Callable = tile_naming_function_default,
                             tile_score_thresh:float = 0.55,
                             tile_scoring_function = scoring_function_1,  
                             level = 0, 
                             save_tiles:bool = False,
                             tiles_folder_path:pathlib.Path=None,
                             return_as_tilesummary_object = True, 
                             wsi_path_to_rois:Dict[pathlib.Path, List[shared.roi.RegionOfInterestPolygon]] = None,
                             minimal_tile_roi_intersection_ratio:float = 1.0,
                             verbose=False, 
                             number_of_processes = None)-> Union[List[TileSummary], pandas.DataFrame]:
    """
    The method WsiToTiles for a list of WSIs/ROIs in parallel.
    
    Arguments:
    wsi_paths: A list of paths to the WSIs or ROIs(=preextracted png files from WSIs)
    shifted_grid: During the process at first all possible tile locations are calculated starting at (0,0) (upper left corner)
                    of the wsi. If shifted grid is True, there will be 6 more grids starting at (1/3*tile_width, 0),
                    (2/3*tile_width, 0), (0, 1/3*tile_height), (0, 2/3*tile_height), (1/3*tile_width, 1/3*tile_height),
                    (2/3*tile_width, 2/3*tile_height)
    tile_heigth: Number of pixels tile height.
    tile_width: Number of pixels tile width.
    minimal_acceptable_tile_height: factor between 0.0 and 1.0. percentage of orig_h ;affects tiles at the edges, which 
                                    cannot have full height 
    minimal_acceptable_tile_width: factor between 0.0 and 1.0. percentage of orig_w ;affects tiles at the edges, which cannot 
                                    have full width
    tile_score_thresh: Tiles with a score higher than the number from "tileScoringFunction" will be saved.
    tile_scoring_function: Function to score one tile to determine if it should be saved or not.
    tile_naming_func: 99% of the time there should be no necessity to change this.
                        A function, that takes a pathlib.Path to the WSI or ROI as an argument and returns a string.
                        This string will then be used as part of the name for the tile (plus some specific tile information and
                        the file format .png, whick is generated by this library).
    level: Level of the WSI you want to extract the tile from. 0 means highest resolution. For not wsi formats like .png leave it at 
            0.
    save_tiles: if True the tiles will be extracted and saved to {tiles_folder_path}
    tiles_folder_path: The folder where the extracted tiles will be saved (only needed if save_tiles=True).
    return_as_tilesummary_object: Set this to true, if you want the TileSummary object and not a pandas dataframe.
    wsi_path_to_rois: a dict with key: wsi_path and value List[shared.roi.RegionOfInterestPolygon]].
    minimal_tile_roi_intersection_ratio: [0.0, 1.0] 
                                         (intersection area between roi and tile)/tile area >= tile_roi_intersection_ratio so 
                                          that a tile will be used for further calculations
    number_of_processes: if None, number_of_processes will be the number of cpu cores
    
    Return:
    if return_as_tilesummary_object == True:
       a List of TileSummary objects will be returned
    else:
        pandas dataframe with coloumns: ['tile_name','wsi_path','level','x_upper_left','y_upper_left','pixels_width','pixels_height']
    """
    
    pbar = tqdm(total=len(wsi_paths))
    results = []
    def update(res):
        results.append(res)
        pbar.update()
        
    def error(e):
        print(e)
    
    with multiprocessing.Pool(processes=number_of_processes) as pool:
        for p in wsi_paths:
            pool.apply_async(WsiToTiles, 
                             kwds={"wsi_path":p,
                                   "shifted_grid":shifted_grid,
                                   "tile_height":tile_height, 
                                   "tile_width":tile_width,
                                   "minimal_acceptable_tile_height":minimal_acceptable_tile_height,
                                   "minimal_acceptable_tile_width":minimal_acceptable_tile_width,
                                   "tile_naming_func":tile_naming_func,
                                   "tile_score_thresh":tile_score_thresh, 
                                   "tile_scoring_function":tile_scoring_function, 
                                   "level":level, 
                                   "save_tiles":save_tiles,
                                   "tiles_folder_path":tiles_folder_path,
                                   "return_as_tilesummary_object":return_as_tilesummary_object, 
                                   "rois":util.safe_dict_access(wsi_path_to_rois, p),
                                   "minimal_tile_roi_intersection_ratio":minimal_tile_roi_intersection_ratio,
                                   "verbose":verbose}, 
                                   callback=update, 
                                   error_callback=error)
            
                
        pool.close()
        pool.join()
    
    if return_as_tilesummary_object:
        return results
    else:
        merged_df = None
        for res in tqdm(results):
            if merged_df is None:
                merged_df = res
            else:
                merged_df = merged_df.append(res, sort=False)
        
        return merged_df.drop_duplicates(inplace=False)
        

       
def wsi_to_scaled_pil_image(wsi_filepath:pathlib.Path, scale_factor = 32, level = 0):
    """
    Convert a WSI training slide to a PIL image.

    Args:

    Returns:

    """
    #wsi = openslide.open_slide(str(wsi_filepath))
    #large_w, large_h = wsi.dimensions
    #new_w = math.floor(large_w / scale_factor)
    #new_h = math.floor(large_h / scale_factor)
    #level = wsi.get_best_level_for_downsample(scale_factor)
    #img = wsi.read_region((0, 0), level, wsi.level_dimensions[level])
    #img = img.convert("RGB")
    #if(scale_factor > 1):
    #    img = img.resize((new_w, new_h), PIL.Image.BILINEAR)
    #return img, large_w, large_h, new_w, new_h

    wsi = openslide.open_slide(str(wsi_filepath))
    large_w, large_h = wsi.level_dimensions[level]
    best_level_for_downsample = wsi.get_best_level_for_downsample(scale_factor)
    new_w, new_h = wsi.level_dimensions[best_level_for_downsample]    
    img = wsi.read_region((0, 0), best_level_for_downsample, wsi.level_dimensions[best_level_for_downsample])
    img = img.convert("RGB")
    return img, large_w, large_h, new_w, new_h, best_level_for_downsample


def create_tilesummary(wsiPath,
                       shifted_grid:bool,
                        tiles_folder_path,
                        img_pil:PIL.Image.Image, 
                        img_pil_filtered:PIL.Image.Image, 
                        wsi_original_width:int, 
                        wsi_original_height:int, 
                        wsi_scaled_width:int, 
                        wsi_scaled_height:int, 
                        tile_height:int, 
                        tile_width:int,
                        minimal_acceptable_tile_height:float,
                        minimal_acceptable_tile_width:float,
                        scale_factor:int,
                        tile_score_thresh:float,
                        tile_scoring_function, 
                        tile_naming_func, 
                        level:int, 
                        best_level_for_downsample:int, 
                        rois:List[shared.roi.RegionOfInterestPolygon] = None,
                        minimal_tile_roi_intersection_ratio:float = 1.0,
                        verbose=False)->TileSummary:
    """
  
    Args:
        wsi_original_width: unscaled width with respect to the specified level
        wsi_original_height: unscaled width with respect to the specified level
        scale_factor: the scale_factor specified by the user
        best_level_for_downsample: result of openslide.OpenSlide.get_best_level_for_downsample(scale_factor)
    """
    
    np_img = util.pil_to_np_rgb(img_pil)
    np_img_filtered = util.pil_to_np_rgb(img_pil_filtered)

    tile_sum = score_tiles(img_np=np_img, 
                           img_np_filtered=np_img_filtered, 
                           wsi_path=wsiPath,
                           shifted_grid=shifted_grid,
                           tiles_folder_path=tiles_folder_path,
                           tile_height=tile_height,
                           tile_width=tile_width,
                           minimal_acceptable_tile_height=minimal_acceptable_tile_height,
                           minimal_acceptable_tile_width=minimal_acceptable_tile_width,
                           scale_factor=scale_factor, 
                           wsi_original_width=wsi_original_width, 
                           wsi_original_height=wsi_original_height, 
                           wsi_scaled_width=wsi_scaled_width, 
                           wsi_scaled_height=wsi_scaled_height,
                           tile_score_thresh=tile_score_thresh,
                           tile_scoring_function=tile_scoring_function, 
                           tile_naming_func=tile_naming_func, 
                           level=level, 
                           best_level_for_downsample=best_level_for_downsample, 
                           rois=rois,
                           minimal_tile_roi_intersection_ratio = minimal_tile_roi_intersection_ratio,
                           verbose=verbose)    
    return tile_sum


def get_num_tiles(rows, cols, row_tile_size, col_tile_size):
    """
    Obtain the number of vertical and horizontal tiles that an image can be divided into given a row tile size and
    a column tile size.
  
    Args:
      rows: Number of rows.
      cols: Number of columns.
      row_tile_size: Number of pixels in a tile row.
      col_tile_size: Number of pixels in a tile column.
  
    Returns:
      Tuple consisting of the number of vertical tiles and the number of horizontal tiles that the image can be divided
      into given the row tile size and the column tile size.
    """
    num_row_tiles = math.ceil(rows / row_tile_size)
    num_col_tiles = math.ceil(cols / col_tile_size)
    return num_row_tiles, num_col_tiles

def get_tile_indices(rows, cols, row_tile_size, col_tile_size, shifted_grid:bool):
    """
    Obtain a list of tile coordinates (starting row, ending row, starting column, ending column, row number, column number).
  
    Args:
      rows: Number of rows. (height in pixels)
      cols: Number of columns. (width in pixels)
      row_tile_size: Number of pixels in a tile row. (tile height in pixels)
      col_tile_size: Number of pixels in a tile column. (tile width in pixels)
      shifted_grid: adds additional shifted grids
  
    Returns:
      List of tuples representing tile coordinates consisting of starting row, ending row,
      starting column, ending column, row number, column number.
  
      row numbers from 1 to rows/row_tile_size (rounded up)
      column numbers from 1 to cols/col_tile_size (rounded up)
    """
    indices = list()
    num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
    for r in range(0, num_row_tiles):
      start_r = r * row_tile_size
      end_r = ((r + 1) * row_tile_size) if (r < num_row_tiles - 1) else rows
      for c in range(0, num_col_tiles):
        start_c = c * col_tile_size
        end_c = ((c + 1) * col_tile_size) if (c < num_col_tiles - 1) else cols
        indices.append((start_r, end_r, start_c, end_c, r + 1, c + 1))
        if(shifted_grid):
            r_third = int(row_tile_size/3)
            c_third = int(col_tile_size/3)
            
            #only rows shifted
            indices.append((start_r+r_third, end_r+r_third, start_c, end_c, r + 1, c + 1))
            indices.append((start_r+2*r_third, end_r+2*r_third, start_c, end_c, r + 1, c + 1))
            
            #only columns shifted
            indices.append((start_r, end_r, start_c+c_third, end_c+c_third, r + 1, c + 1))
            indices.append((start_r, end_r, start_c+2*c_third, end_c+2*c_third, r + 1, c + 1))
        
            #rows and columns shifted
            indices.append((start_r+r_third, end_r+r_third, start_c+c_third, end_c+c_third, r + 1, c + 1))
            indices.append((start_r+2*r_third, end_r+2*r_third, start_c+2*c_third, end_c+2*c_third, r + 1, c + 1))
    return indices

def tile_to_pil_tile(tile:Tile):
      """
      Convert tile information into the corresponding tile as a PIL image read from the whole-slide image file.

      Args:
        tile: Tile object.

      Return:
        Tile as a PIL image.
      """
      return ExtractTileFromWSI(tile.tile_summary.wsi_path, 
                                tile.get_x(), 
                                tile.get_y(), 
                                tile.get_width(), 
                                tile.get_height(), 
                                tile.level)


def tile_to_np_tile(tile):
  """
  Convert tile information into the corresponding tile as a NumPy image read from the whole-slide image file.

  Args:
    tile: Tile object.

  Return:
    Tile as a NumPy image.
  """
  pil_img = tile_to_pil_tile(tile)
  np_img = util.pil_to_np_rgb(pil_img)
  return np_img



def get_tile_image_path(tile:Tile):
  """
  Obtain tile image path based on tile information such as row, column, row pixel position, column pixel position,
  pixel width, and pixel height.

  Args:
    tile: Tile object.

  Returns:
    Path to image tile.
  """
  t = tile
  if tile.tiles_folder_path is None:
      return os.path.join(tile.tile_naming_func(tile.tile_summary.wsi_path) + "-" + 'tile' + "-r%d-c%d-x%d-y%d-w%d-h%d" % (
                             t.r, t.c, t.o_c_s, t.o_r_s, t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s) + "." + 'png')
  else:
      return os.path.join(tile.tiles_folder_path, 
                          tile.tile_naming_func(tile.tile_summary.wsi_path) + "-" + 'tile' + "-r%d-c%d-x%d-y%d-w%d-h%d" % (
                             t.r, t.c, t.o_c_s, t.o_r_s, t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s) + "." + 'png') 


def save_display_tile(tile, save, display):
  """
  Save and/or display a tile image.

  Args:
    tile: Tile object.
    save: If True, save tile image.
    display: If True, dispaly tile image.
  """
  tile_pil_img = tile_to_pil_tile(tile)

  if save:
    t = Time()
    img_path = get_tile_image_path(tile)
    dir = os.path.dirname(img_path)
    if not os.path.exists(dir):
      os.makedirs(dir)
    tile_pil_img.save(img_path)
    tile.tile_path = img_path
    #print("%-20s | Time: %-14s  Name: %s" % ("Save Tile", str(t.elapsed()), img_path))

  if display:
    tile_pil_img.show()

def get_rois_the_given_tile_is_in(rois:List[shared.roi.RegionOfInterestPolygon], 
                               tile:shapely.geometry.Polygon, 
                               minimal_tile_roi_intersection_ratio:float)->List[shared.roi.RegionOfInterestPolygon]:
    containing_rois = []
    for roi in rois:
        if((roi.polygon.intersection(tile).area/tile.area) >= minimal_tile_roi_intersection_ratio):
            containing_rois.append(roi)
    return containing_rois

def score_tiles(img_np:np.array, 
                img_np_filtered:np.array, 
                wsi_path:pathlib.Path,
                shifted_grid:bool,
                tiles_folder_path:pathlib.Path,
                tile_height:int, 
                tile_width:int,
                minimal_acceptable_tile_height:float,
                minimal_acceptable_tile_width:float,
                scale_factor:int, 
                wsi_original_width:int, 
                wsi_original_height:int, 
                wsi_scaled_width:int, 
                wsi_scaled_height:int,
                tile_score_thresh:float,
                tile_scoring_function, 
                tile_naming_func, 
                level:int, 
                best_level_for_downsample:int, 
                rois:List[shared.roi.RegionOfInterestPolygon]=None,
                minimal_tile_roi_intersection_ratio:float = 1.0,
                verbose=False) -> TileSummary:
    """
    Scores all tiles for a slide and returns the results in a TileSummary object.
    If regions of interests are specified, only tiles within those regions will be scored to reduce processing time.
    Tiles within those regions will be cut to fit exactly into the region. So there will be tiles that are smaller than 
    the specified height and width.

    Args:
        wsi_original_width: unscaled width with respect to the specified level
        wsi_original_height: unscaled width with respect to the specified level
        scale_factor: the scale_factor specified by the user
        best_level_for_downsample: result of openslide.OpenSlide.get_best_level_for_downsample(scale_factor) 


    Returns:
    TileSummary object which includes a list of Tile objects containing information about each tile.
    """
    
    #if no rois are specified, just create one "fake" roi, that is as big as the whole image
    if(rois is None or len(rois) == 0):
        rois = [shared.roi.RegionOfInterestPolygon(roi_id = wsi_path,
                                          vertices = np.array([[0,0], [0, wsi_original_height], \
                                                               [wsi_original_width, wsi_original_height], \
                                                               [wsi_original_width, 0]]),
                                          level = level)]
    
    
    #print(rois)
    
    # if roi level and tile extraction level differ, adjust roi dimensions to the tile level 
    # e.g. roi level is 0, tile level is 2 -> divide roi dimensions by 2^(2-0)
    #for roi in rois:       
    #    if roi.level != level:
    #        roi.change_level_in_place(level)
    
    
    real_scale_factor = int(math.pow(2,best_level_for_downsample-level))
    tile_height_scaled = util.adjust_level(tile_height, level, best_level_for_downsample)
    tile_width_scaled = util.adjust_level(tile_width, level, best_level_for_downsample)

    num_row_tiles, num_col_tiles = get_num_tiles(wsi_scaled_height, 
                                                 wsi_scaled_width, 
                                                 tile_height_scaled, 
                                                 tile_width_scaled)

    tile_sum = TileSummary(wsi_path=wsi_path,
                           tiles_folder_path=tiles_folder_path,
                             orig_w=wsi_original_width,
                             orig_h=wsi_original_height,
                             orig_tile_w=tile_width,
                             orig_tile_h=tile_height,
                             minimal_acceptable_tile_height = minimal_acceptable_tile_height,
                             minimal_acceptable_tile_width = minimal_acceptable_tile_width,
                             scale_factor=scale_factor,
                             scaled_w=wsi_scaled_width,
                             scaled_h=wsi_scaled_height,
                             scaled_tile_w=tile_width_scaled,
                             scaled_tile_h=tile_height_scaled,
                             tissue_percentage=filter.tissue_percent(img_np_filtered),
                             num_col_tiles=num_col_tiles,
                             num_row_tiles=num_row_tiles,
                             tile_score_thresh=tile_score_thresh,
                             level=level,
                             best_level_for_downsample=best_level_for_downsample,
                             real_scale_factor=real_scale_factor, 
                             rois=rois)   
    


    count = 0
    high = 0
    medium = 0
    low = 0
    none = 0
    
    
    #rois_downsample_level = []
    #for roi in rois:
    #    rois_downsample_level.append(roi.change_level_deep_copy(new_level=best_level_for_downsample))
    
    tile_indices = get_tile_indices(wsi_scaled_height, wsi_scaled_width, tile_height_scaled, tile_width_scaled, shifted_grid)
    for ti in tile_indices:
        #coordinates with respect to upper left point of wsi as (0,0) on the level chosen for downsampling
        #r_s: row_start, r_e: row_end  (pixel values in y-dimension)
        #c_s: column_start, c_e: column_end (pixel values in x-dimension)
        #r: row number (starting at 1 and number of rows is
        #heigt of the image divided by tile's height)
        #c: column number (starting at 1 and number of columns is
        #width of the image divided by tile's width)
        r_s, r_e, c_s, c_e, r, c = ti
                
        #o_c_s: original_column_start, etc. pixel values with respect to the desired tile_extraction level
        o_c_s, o_r_s = slide.small_to_large_mapping((c_s, r_s), (wsi_original_width, wsi_original_height), real_scale_factor)
        #print("o_c_s: " + str(o_c_s))
        #print("o_r_s: " + str(o_r_s))
        o_c_e, o_r_e = slide.small_to_large_mapping((c_e, r_e), (wsi_original_width, wsi_original_height), real_scale_factor)
        #print("o_c_e: " + str(o_c_e))
        #print("o_r_e: " + str(o_r_e))

        # pixel adjustment in case tile dimension too large (for example, 1025 instead of 1024)
        if (o_c_e - o_c_s) > tile_width:
            o_c_e -= 1
        if (o_r_e - o_r_s) > tile_height:
            o_r_e -= 1

        
        tile_polygon = shapely.geometry.box(minx=o_c_s, miny=o_r_s, maxx=o_c_e, maxy=o_r_e)
        rois_the_current_tile_is_in = get_rois_the_given_tile_is_in(rois=rois, 
                                                 tile=tile_polygon, 
                                                 minimal_tile_roi_intersection_ratio=minimal_tile_roi_intersection_ratio)
        # short circuit here, if the tile is not in one of the rois
        if(len(rois_the_current_tile_is_in) == 0):
            continue
                                
        roi = rois_the_current_tile_is_in[0]     
        count += 1  # tile_num            
            
            
            
        np_scaled_filtered_tile = img_np_filtered[int(r_s):int(r_e), int(c_s):int(c_e)]
        t_p = filter.tissue_percent(np_scaled_filtered_tile)
        amount = tissue_quantity(t_p)
        if amount == TissueQuantity.HIGH:
            high += 1
        elif amount == TissueQuantity.MEDIUM:
            medium += 1
        elif amount == TissueQuantity.LOW:
            low += 1
        elif amount == TissueQuantity.NONE:
            none += 1

        score, color_factor, s_and_v_factor, quantity_factor = score_tile(np_scaled_filtered_tile, t_p, r, c, 
                                                                              tile_scoring_function)
       
        tile = Tile(tile_summary=tile_sum, 
                         tiles_folder_path=tiles_folder_path, 
                         np_scaled_filtered_tile=np_scaled_filtered_tile, 
                         tile_num = count, 
                         r=r, 
                         c=c, 
                         r_s=r_s, 
                         r_e=r_e, 
                         c_s=c_s, 
                         c_e=c_e, 
                         o_r_s=o_r_s, 
                         o_r_e=o_r_e, 
                         o_c_s=o_c_s, 
                         o_c_e=o_c_e, 
                         t_p=t_p, 
                         color_factor=color_factor, 
                         s_and_v_factor=s_and_v_factor, 
                         quantity_factor=quantity_factor, 
                         score=score, 
                         tile_naming_func=tile_naming_func, 
                         level=level, 
                         best_level_for_downsample=best_level_for_downsample, 
                         real_scale_factor=real_scale_factor, 
                         roi=roi)
            
        tile_sum.tiles.append(tile)


    tile_sum.count = count
    tile_sum.high = high
    tile_sum.medium = medium
    tile_sum.low = low
    tile_sum.none = none
      
    tiles_by_score = tile_sum.tiles_by_score()
    rank = 0
    for t in tiles_by_score:
        rank += 1
        t.rank = rank

    return tile_sum



def score_tile(np_tile, tissue_percent, row, col, scoring_function):
    """
    Score tile based on tissue percentage, color factor, saturation/value factor, and tissue quantity factor.
    
    Args:
    np_tile: Tile as NumPy array.
    tissue_percent: The percentage of the tile judged to be tissue.
    slide_num: Slide number.
    row: Tile row.
    col: Tile column.
    
    Returns tuple consisting of score, color factor, saturation/value factor, and tissue quantity factor.
    """
    color_factor = hsv_purple_pink_factor(np_tile)
    s_and_v_factor = hsv_saturation_and_value_factor(np_tile)
    amount = tissue_quantity(tissue_percent)
    quantity_factor = tissue_quantity_factor(amount)
    combined_factor = color_factor * s_and_v_factor   
    score = scoring_function(tissue_percent, combined_factor)
    
    #if combined_factor != 0.0 or tissue_percent != 0.0:
     #   print(f'before: {score}')            
                
    # scale score to between 0 and 1
    score = 1.0 - (10.0 / (10.0 + score))
    
    #if combined_factor != 0.0 or tissue_percent != 0.0:
      #  print(f'after: {score}') 
                  
    return score, color_factor, s_and_v_factor, quantity_factor

def tissue_quantity_factor(amount):
  """
  Obtain a scoring factor based on the quantity of tissue in a tile.

  Args:
    amount: Tissue amount as a TissueQuantity enum value.

  Returns:
    Scoring factor based on the tile tissue quantity.
  """
  if amount == TissueQuantity.HIGH:
    quantity_factor = 1.0
  elif amount == TissueQuantity.MEDIUM:
    quantity_factor = 0.2
  elif amount == TissueQuantity.LOW:
    quantity_factor = 0.1
  else:
    quantity_factor = 0.0
  return quantity_factor


def tissue_quantity(tissue_percentage):
  """
  Obtain TissueQuantity enum member (HIGH, MEDIUM, LOW, or NONE) for corresponding tissue percentage.

  Args:
    tissue_percentage: The tile tissue percentage.

  Returns:
    TissueQuantity enum member (HIGH, MEDIUM, LOW, or NONE).
  """
  if tissue_percentage >= TISSUE_HIGH_THRESH:
    return TissueQuantity.HIGH
  elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
    return TissueQuantity.MEDIUM
  elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
    return TissueQuantity.LOW
  else:
    return TissueQuantity.NONE



def rgb_to_hues(rgb):
  """
  Convert RGB NumPy array to 1-dimensional array of hue values (HSV H values in degrees).

  Args:
    rgb: RGB image as a NumPy array

  Returns:
    1-dimensional array of hue values in degrees
  """
  hsv = filter.filter_rgb_to_hsv(rgb, display_np_info=False)
  h = filter.filter_hsv_to_h(hsv, display_np_info=False)
  return h


def hsv_saturation_and_value_factor(rgb):
  """
  Function to reduce scores of tiles with narrow HSV saturations and values since saturation and value standard
  deviations should be relatively broad if the tile contains significant tissue.

  Example of a blurred tile that should not be ranked as a top tile:
    ../data/tiles_png/006/TUPAC-TR-006-tile-r58-c3-x2048-y58369-w1024-h1024.png

  Args:
    rgb: RGB image as a NumPy array

  Returns:
    Saturation and value factor, where 1 is no effect and less than 1 means the standard deviations of saturation and
    value are relatively small.
  """
  hsv = filter.filter_rgb_to_hsv(rgb, display_np_info=False)
  s = filter.filter_hsv_to_s(hsv)
  v = filter.filter_hsv_to_v(hsv)
  s_std = np.std(s)
  v_std = np.std(v)
  if s_std < 0.05 and v_std < 0.05:
    factor = 0.4
  elif s_std < 0.05:
    factor = 0.7
  elif v_std < 0.05:
    factor = 0.7
  else:
    factor = 1

  factor = factor ** 2
  return factor


def hsv_purple_deviation(hsv_hues):
  """
  Obtain the deviation from the HSV hue for purple.

  Args:
    hsv_hues: NumPy array of HSV hue values.

  Returns:
    The HSV purple deviation.
  """
  purple_deviation = np.sqrt(np.mean(np.abs(hsv_hues - HSV_PURPLE) ** 2))
  return purple_deviation


def hsv_pink_deviation(hsv_hues):
  """
  Obtain the deviation from the HSV hue for pink.

  Args:
    hsv_hues: NumPy array of HSV hue values.

  Returns:
    The HSV pink deviation.
  """
  pink_deviation = np.sqrt(np.mean(np.abs(hsv_hues - HSV_PINK) ** 2))
  return pink_deviation


def hsv_purple_pink_factor(rgb):
  """
  Compute scoring factor based on purple and pink HSV hue deviations and degree to which a narrowed hue color range
  average is purple versus pink.

  Args:
    rgb: Image an NumPy array.

  Returns:
    Factor that favors purple (hematoxylin stained) tissue over pink (eosin stained) tissue.
  """
  hues = rgb_to_hues(rgb)
  hues = hues[hues >= 260]  # exclude hues under 260
  hues = hues[hues <= 340]  # exclude hues over 340
  if len(hues) == 0:
    return 0  # if no hues between 260 and 340, then not purple or pink
  pu_dev = hsv_purple_deviation(hues)
  pi_dev = hsv_pink_deviation(hues)
  avg_factor = (340 - np.average(hues)) ** 2

  if pu_dev == 0:  # avoid divide by zero if tile has no tissue
    return 0

  factor = pi_dev / pu_dev * avg_factor
  return factor


def hsv_purple_vs_pink_average_factor(rgb, tissue_percentage):
  """
  Function to favor purple (hematoxylin) over pink (eosin) staining based on the distance of the HSV hue average
  from purple and pink.

  Args:
    rgb: Image as RGB NumPy array
    tissue_percentage: Amount of tissue on the tile

  Returns:
    Factor, where >1 to boost purple slide scores, <1 to reduce pink slide scores, or 1 no effect.
  """

  factor = 1
  # only applies to slides with a high quantity of tissue
  if tissue_percentage < TISSUE_HIGH_THRESH:
    return factor

  hues = rgb_to_hues(rgb)
  hues = hues[hues >= 200]  # Remove hues under 200
  if len(hues) == 0:
    return factor
  avg = np.average(hues)
  # pil_hue_histogram(hues).show()

  pu = HSV_PURPLE - avg
  pi = HSV_PINK - avg
  pupi = pu + pi
  # print("Av: %4d, Pu: %4d, Pi: %4d, PuPi: %4d" % (avg, pu, pi, pupi))
  # Av:  250, Pu:   20, Pi:   80, PuPi:  100
  # Av:  260, Pu:   10, Pi:   70, PuPi:   80
  # Av:  270, Pu:    0, Pi:   60, PuPi:   60 ** PURPLE
  # Av:  280, Pu:  -10, Pi:   50, PuPi:   40
  # Av:  290, Pu:  -20, Pi:   40, PuPi:   20
  # Av:  300, Pu:  -30, Pi:   30, PuPi:    0
  # Av:  310, Pu:  -40, Pi:   20, PuPi:  -20
  # Av:  320, Pu:  -50, Pi:   10, PuPi:  -40
  # Av:  330, Pu:  -60, Pi:    0, PuPi:  -60 ** PINK
  # Av:  340, Pu:  -70, Pi:  -10, PuPi:  -80
  # Av:  350, Pu:  -80, Pi:  -20, PuPi: -100

  if pupi > 30:
    factor *= 1.2
  if pupi < -30:
    factor *= .8
  if pupi > 0:
    factor *= 1.2
  if pupi > 50:
    factor *= 1.2
  if pupi < -60:
    factor *= .8

  return factor