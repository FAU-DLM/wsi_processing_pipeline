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


#https://stackoverflow.com/questions/46641078/how-to-avoid-circular-dependency-caused-by-type-hinting-of-pointer-attributes-in
from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from .shared.tile import Tile


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
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
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


import tile_extraction
from tile_extraction import util
from util import adjust_level, safe_dict_access, pil_to_np_rgb, show_wsi_with_rois
import filter, slide, openslide_overwrite
import shared
from shared import roi
#from shared.tile import Tile ## see future import
from shared.roi import *
from shared.enums import DatasetType, TissueQuantity





#TISSUE_HIGH_THRESH = 80
#TISSUE_LOW_THRESH = 10
HSV_PURPLE = 270
HSV_PINK = 330

############################# classes #########################################


class Vertex:
    def __init__(self, x:float, y:float):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f'(x:{self.x}, y:{self.y})'
    
    def __str__(self):
        return f'(x:{self.x}, y:{self.y})'
    
    def __add__(self, o)->Vertex:
        
        #print(type(o))
        
        if(type(o) is np.ndarray and (o.shape == (2,) or o.shape == (2,1))):
            self.x += o[0]
            self.y += o[1]
        elif(type(o) is Vertex or type(o) is __main__.Vertex):
            self.x += o.x
            self.y += o.y
        else:
            raise TypeError(f'Vertex class does not support addition with type: {type(o)}')
            
        return self

    def __sub__(self, o):
        return self.__add__(copy.deepcopy(o)*(-1))
                  
    def __mul__(self, o):
        if(type(o) is int or type(o) is float):
            self.x *= o
            self.y *= o
        else:
            raise TypeError(f'Vertex class does not support multiplication with type: {type(o)}')
            
        return self
            
    def __rmatmul__(self, o):
        if(type(o) is np.ndarray):
            return o@np.array([self.x, self.y])
        else:
            raise TypeError(f'Vertex class does not support (right sided) \
            matrix multiplication with type: {type(o)}')
            
        return self
    
    def __call__(self):
        return np.array([self.x, self.y])
        
    def rotate_around_pivot(self, angle:float, pivot = np.array([0, 0])):
        """
        Rotates itself clockwise around the specified pivot with the specified angle.
        """
        self.__add__(-pivot)
        radians = math.radians(angle)
        rotation_matrix = np.array([[math.cos(radians), -math.sin(radians)],\
                                    [math.sin(radians), math.cos(radians)]])
        new_coordinates = rotation_matrix@self.__call__()
        self.x = new_coordinates[0]
        self.y = new_coordinates[1]
        self.__add__(pivot)
        
        return self
    
    def deepcopy(self):
        return Vertex(x=self.x, y=self.y)
    
    def change_level(self, current_level:int, new_level:int):
        """
        Arguments:
        
        Return:
        """
        self.x = adjust_level(value_to_adjust=self.x, from_level=current_level, to_level=new_level)
        self.y = adjust_level(value_to_adjust=self.y, from_level=current_level, to_level=new_level)
        return self

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
    
    def __add__(self, o:Union[np.ndarray, Vertex]):
        self.ul + o
        self.ur + o
        self.lr + o
        self.ll + o
        return self
        
    def __sub__(self, o:Union[np.ndarray, Vertex]):
        return self.__add__(copy.deepcopy(o)*(-1))
        
    def __mul__(self, o:int):
        self.ul * o
        self.ur * o
        self.lr * o
        self.ll * o
        return self
    
    def __rotate_all_vertices(self, angle:float, pivot:np.ndarray):
        self.ul.rotate_around_pivot(angle=angle, pivot=pivot)
        self.ur.rotate_around_pivot(angle=angle, pivot=pivot)
        self.lr.rotate_around_pivot(angle=angle, pivot=pivot)
        self.ll.rotate_around_pivot(angle=angle, pivot=pivot)
        return self
    
    def deepcopy(self):
        ul_dc = copy.deepcopy(self.ul)
        ur_dc = copy.deepcopy(self.ur)
        lr_dc = copy.deepcopy(self.lr)
        ll_dc = copy.deepcopy(self.ll)
        return Rectangle(ul_dc,ur_dc,lr_dc,ll_dc)
    
    def polygon(self)->shapely.geometry.Polygon:
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
        return self
        
    def rotate_around_pivot_without_orientation_change(self, 
                                                       angle:float, 
                                                       pivot = np.array([0,0])):
        """
        Rotates the rectangle's centroid around the specified pivot.
        The orientations of the edges do not change. 
        Arguments:
            angle: degrees clockwise
        """
        self.__add__(-pivot)
        
        centroid_old = Vertex(x=self.polygon().centroid.x, y=self.polygon().centroid.y)
        centroid_new = copy.deepcopy(centroid_old)
        centroid_new.rotate_around_pivot(angle=angle, pivot=np.array([0,0]))
        self.__add__(-centroid_old())
        self.__add__(centroid_new())
        
        self.__add__(pivot)
        return self
    
    def rotate(self, angle:float, pivot = np.array([0,0])):
        """
        Combines rotate_around_itself and rotate_around_pivot
        Arguments:
            angle: degrees clockwise
        """
        self.rotate_around_itself(angle=angle)
        self.rotate_around_pivot_without_orientation_change(angle=angle, pivot=pivot)
        return self
    
    def as_roi(self, level)->shared.roi.RegionOfInterestPolygon:
        """
        Creates and returns a RegionOfInterestPolygon from its values.
        Arguments:
            level: WSI level
        """
        return shared.roi.RegionOfInterestPolygon(roi_id=self.__str__(), 
                                                  vertices=self.__call__(), 
                                                  level = level)
    
    def as_shapely_polygon(self)->shapely.geometry.Polygon:
        ul = (self.ul.x, self.ul.y)
        ur = (self.ur.x, self.ur.y)
        lr = (self.lr.x, self.lr.y)
        ll = (self.ll.x, self.ll.y)
        return shapely.geometry.Polygon(shell=[ul, ur, lr, ll])
    
    def get_outer_bounds(self):
        """
        returns a new Rectangle which contains this Rectangle but its edges are parallel to the 
        WSI. So if this Rectangle is not rotated, the new Rectangle will have the same
        spatial information. But if this Rectangle is rotated the new Rectangle will be larger.
        """
        p = self.as_shapely_polygon()
        b_ul = Vertex(x=p.bounds[0], y=p.bounds[1])
        b_ur = Vertex(x=p.bounds[2], y=p.bounds[1])
        b_lr = Vertex(x=p.bounds[2], y=p.bounds[3])
        b_ll = Vertex(x=p.bounds[0], y=p.bounds[3])
        return Rectangle(upper_left=b_ul, 
                         upper_right=b_ur, 
                         lower_right=b_lr, 
                         lower_left=b_ll)
    
    def is_rotated(self)->bool:
        return self.ul.y != self.ur.y
    
    def width(self)->float:
        if(self.is_rotated()):
            a = self.ur.x - self.ul.x
            b = self.ur.y - self.ul.y
            return math.sqrt(a**2 + b**2)
            
        else:
            return self.ur.x - self.ul.x
    
    def height(self)->float:
        if(self.is_rotated()):
            a = self.ll.y - self.ul.y
            b = self.ul.x - self.ll.x
            return math.sqrt(a**2 + b**2)
            
        else:
            return self.ll.y - self.ul.y
        
    def change_level(self, current_level:int, new_level:int):
        """
        Arguments:
        
        Return:
        """
        self.ul.change_level(current_level=current_level, new_level=new_level)
        self.ur.change_level(current_level=current_level, new_level=new_level)
        self.lr.change_level(current_level=current_level, new_level=new_level)
        self.ll.change_level(current_level=current_level, new_level=new_level)
        return self
        
class Grid:
    def __init__(self, 
                 min_width:int, 
                 min_height:int, 
                 tile_width:int, 
                 tile_height:int,
                 level:int,
                 coordinate_origin_x:int = 0, 
                 coordinate_origin_y:int = 0, 
                 angle:int = 0):
        """
        level: WSI level
        """
        ##
        #init object attributes
        ##
        self.min_width = min_width
        self.min_height = min_height
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.level = level
        self.coordinate_origin_x = coordinate_origin_x
        self.coordinate_origin_y = coordinate_origin_y
        
        self.grid_centroid_float = np.array([coordinate_origin_x+min_width/2,\
                                             coordinate_origin_y+min_height/2])
        self.grid_centroid_int = np.array([int(coordinate_origin_x+min_width/2),\
                                           int(coordinate_origin_y+min_height/2)])
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
                return rect.as_roi(level=self.level)
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
                    minimal_intersection_quotient:float):
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
                    rect_as_roi = elem.as_roi(level=self.level)
                    try:
                        intersection_area = roi.polygon.intersection(rect_as_roi.polygon).area
                    except shapely.geos.TopologicalError as e:
                        #possible temporary fix could be "roi.polygon.buffer(0).intersection(rect_as_roi.polygon).area"
                        #as described here: https://github.com/gboeing/osmnx/issues/278
                        #but buffer(0) migth change the roi in an unexpected way
                        #intersection_area = roi.polygon.buffer(0).intersection(rect_as_roi.polygon).area
                        
                        #print(f'method tiles.Grid.filter_grid: {e}')
                        self.grid[row][col] = None
                        continue
                                                                    
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
                 grids_per_roi:int = 1, 
                 level:int = 0):
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
            level: WSI level; 0 means highest resolution and magnification.
                   The GridManager checks each roi, if its level matches the given and if that's not
                    the case changes it. It works on deep copies of the rois.
                    Tile_width and tile_height are handled independently 
                    from the given level.
        """

        if(rois is not None):
            rois_dc = [copy.deepcopy(r) for r in rois]
            shared.roi.merge_overlapping_rois(rois=rois_dc)
            [r.change_level_in_place(new_level=level) for r in rois_dc if r.level != level]
            self.rois = rois_dc
        
        self.wsi_path = wsi_path
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.grids_per_roi = grids_per_roi
        self.level = level
        
        self.grids = []
        self.roi_to_grids = {}
        self.grid_to_roi = {}
        
        # if there is no roi given, one roi that spans the complete 
        # WSI is created
        if(rois is None or len(rois) == 0):
            w = slide.open_slide(path=wsi_path)
            wsi_width = w.level_dimensions[level][0]
            wsi_height = w.level_dimensions[level][1]
            ul = np.array([0,0])
            ur = np.array([wsi_width, 0])
            lr = np.array([wsi_width, wsi_height])
            ll = np.array([0, wsi_height])
            vertices = np.array([ul, ur, lr, ll])
            r = RegionOfInterestPolygon(roi_id=f'{wsi_path.stem} - dummy_roi', 
                                        vertices=vertices, level=level)
            self.rois = [r]
            for i in range(grids_per_roi):
                shift_origin_x = tile_width*i/grids_per_roi
                shift_origin_y = tile_height*i/grids_per_roi
                g = Grid(min_width=w.level_dimensions[level][0], 
                         min_height=w.dimensions[1], 
                         tile_width=tile_width, 
                         tile_height=tile_height,
                         level=level,
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
                             level=level,
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
            show_wsi_with_rois(wsi_path=self.wsi_path, 
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
        Trys different angles from 0° - 90° with intervals of the size "stepsize" to find the best angle, 
        to fit in the most tiles.
        It also filters out tiles which are not inside rois.
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
        
    def get_all_rectangles(self)->List[Rectangle]:
        return [r for g in self.grids for r in g.get_rectangles()]
    
    def get_rois_the_given_tile_is_in(self,
                                      rect_tile:Rectangle,
                                      rect_level:int,
                                    minimal_tile_roi_intersection_ratio:float)\
                                ->List[shared.roi.RegionOfInterestPolygon]:
        """
        rect_level: wsi level
        """
        containing_rois = []
        for roi in self.rois:
            if(roi.level != rect_level):
                rect_tile = rect_tile.deepcopy().change_level(current_level=rect_level, 
                                                              new_level=roi.level)
                rect_level = roi.level
            try:
                if((roi.polygon.intersection(rect_tile.polygon()).area/rect_tile.polygon().area)\
                       >= minimal_tile_roi_intersection_ratio):
                            containing_rois.append(roi)
            except Exception as e:
                print('Excpetion in Method "get_rois_the_given_tile_is_in"')
                pass
                
        return containing_rois
    
    
#from pythonlangutil.overload import Overload, signature

import cv2
def pil_to_open_cv(pil_image):
    return cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)

def open_cv_to_pil(open_cv_image):
    return PIL.Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))

class WsiHandler:
    def __init__(self, wsi_path:Union[str, pathlib.Path]):
        self.wsi_path = pathlib.Path(wsi_path)
        self.slide = slide.open_slide(str(self.wsi_path))
    
    #@Overload
    #@signature("int", "int", "int", "int", "int")
    def extract_tile_from_wsi_1(self, 
                               ul_x:int, 
                               ul_y:int, 
                               width:int, 
                               height:int, 
                               level:int)->PIL.Image:
        """
        Args:
            ul_x: x-coordinate of the upper left pixel. The method assumes, 
                that you know the dimensions of your specified level.
            ul_y: y-coordinate of the upper left pixel. The method assumes, 
                that you know the dimensions of your specified level.
            width: tile width
            height: tile height
            level: Level of the WSI you want to extract the tile from. 
                    0 means highest resolution.
            
        Return:
            tile as PIL.Image as RGB
        """
        s = self.slide
        wsi_width = s.level_dimensions[level][0]
        wsi_height = s.level_dimensions[level][1]
        
        #read_region() expects the coordinates of the upper left pixel with respect to level 0
        ul_x_level_0 = adjust_level(value_to_adjust=ul_x, from_level=level, to_level=0)
        ul_y_level_0 = adjust_level(value_to_adjust=ul_y, from_level=level, to_level=0)
        
        ul_x_level_0 = int(ul_x_level_0)
        ul_y_level_0 = int(ul_y_level_0)
        width = int(width)
        height = int(height)
        
        tile_region = s.read_region((ul_x_level_0, ul_y_level_0), level, (width, height))
        # RGBA to RGB
        pil_img = tile_region.convert("RGB")
        return pil_img
    
    #@extract_tile_from_wsi.overload
    #@signature("Rectangle")
    def extract_tile_from_wsi_2(self, 
                                rectangle_tile:Rectangle, 
                                level:int)->PIL.Image:
        """
        On how to extract a rotated rectangle:
        https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/
        """
        
        # check if rectangle is rotated
        if(not rectangle_tile.is_rotated()):
            width = rectangle_tile.ur.x - rectangle_tile.ul.x
            height = rectangle_tile.ll.y - rectangle_tile.ul.y
            return self.extract_tile_from_wsi_1(ul_x=rectangle_tile.ul.x, 
                                         ul_y=rectangle_tile.ul.y, 
                                         width=width, 
                                         height=height, 
                                         level=level)
        
        
        else:
            rect_bounds = rectangle_tile.get_outer_bounds()
            bounds_pil = self.extract_tile_from_wsi_1(ul_x=int(rect_bounds.ul.x), 
                                                    ul_y=int(rect_bounds.ul.y), 
                                                    width=int(rect_bounds.ur.x - rect_bounds.ul.x), 
                                                    height=int(rect_bounds.ll.y - rect_bounds.ul.y), 
                                                    level=level)
            rect_tile_deepcopy = rectangle_tile.deepcopy()
            rect_bounds_deepcopy = rect_bounds.deepcopy()
            rect_tile_adjusted_origin = rect_tile_deepcopy - rect_bounds_deepcopy.ul
            
            cnt = np.array([[[int(rect_tile_adjusted_origin.ul.x), int(rect_tile_adjusted_origin.ul.y)]],
                   [[int(rect_tile_adjusted_origin.ur.x), int(rect_tile_adjusted_origin.ur.y)]],
                   [[int(rect_tile_adjusted_origin.lr.x), int(rect_tile_adjusted_origin.lr.y)]],
                   [[int(rect_tile_adjusted_origin.ll.x), int(rect_tile_adjusted_origin.ll.y)]]])
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            width = int(rect[1][0])
            height = int(rect[1][1])
            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height+1],
                                [0, 0],
                                [width+1, 0],
                                [width+1, height+1]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(pil_to_open_cv(bounds_pil), M, (width+1, height+1))
            return open_cv_to_pil(warped)
        
    def get_wsi_as_pil_image(self, level = 5):
        wsi = openslide.open_slide(str(self.wsi_path))
        large_w, large_h = wsi.level_dimensions[level]
        #best_level_for_downsample = wsi.get_best_level_for_downsample(scale_factor)
        new_w, new_h = wsi.level_dimensions[level]    
        pil_img = wsi.read_region((0, 0), level, wsi.level_dimensions[level])
        pil_img = pil_img.convert("RGB")
        return pil_img

        
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
      
    scale_factor = None #for faster processing the wsi is scaled down internally, 
                        #the resulting tiles are on maximum resolution
                        #depending on the specified level
    scaled_w = None 
    scaled_h = None
    scaled_tile_w = None
    scaled_tile_h = None
    #mask_percentage = None
    tile_score_thresh = None
    level = None
    best_level_for_downsample = None
    real_scale_factor = None        
    rois:RegionOfInterest   
    grid_manager = None   
    tiles = None

    def __init__(self, 
                 wsi_path,
                 tiles_folder_path, 
                 orig_w, 
                 orig_h, 
                 orig_tile_w, 
                 orig_tile_h,
                 scale_factor, 
                 scaled_w, 
                 scaled_h, 
                 scaled_tile_w,
                 scaled_tile_h, 
                 #tissue_percentage, 
                 tile_score_thresh, 
                 level, 
                 best_level_for_downsample,
                 real_scale_factor, 
                 rois:List[RegionOfInterest], 
                 grid_manager:GridManager):

        """
        Arguments:
            level: whole-slide image's level, the tiles shall be extracted from
            orig_w, orig_h: original height and original width depend on the specified level. 
                            With each level, the dimensions half.                                                            
                                                                       
            scale_factor:  Downscaling is applied during tile calculation to speed up the process. 
                           The tiles in the end get extracted from the full resolution. 
                           The full resolution depends on the level, the user specifies. 
                            The higher the level, the lower the resolution/magnification.
                            Therefore less downsampling needs to be applied during tile calculation, 
                            to achieve same speed up.
                            So e.g. the wsi has dimensions of 10000x10000 pixels on level 0. 
                            A scale_factor of 32 is speficied. 
                            Then calculations will be applied on
                            a downscaled version of the wsi with dimensions on the level log2(32)

            real_scale_factor: if a scale_factor of e.g. 32 is specified and a level of 0, 
                                from which the tiles shall be extracted, 
                                scale_factor==real_scale_factor.
                                 For each level, the wsi dimensions half.
                                 That means for a scale_factor of 32 
                                 and level 1 the real_scale_factor would be only 16.
                                 downscaling is applied during tile calculation to speed up the process. 
                                 The tile in the end get extracted from the full resolution
                                 The full resolution depends on the level, the user specifies. 
                                 The higher the level, the lower the 
                                 resolution/magnification.
                                 Therefore less downsampling needs to be applied during tile calculation, 
                                 to achieve same speed up.

            best_level_for_downsample: result of openslide.OpenSlide.get_best_level_for_downsample(scale_factor)
        """

        self.wsi_path = wsi_path
        self.tiles_folder_path = tiles_folder_path
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.orig_tile_w = orig_tile_w
        self.orig_tile_h = orig_tile_h          
        self.scale_factor = scale_factor
        self.scaled_w = scaled_w
        self.scaled_h = scaled_h
        self.scaled_tile_w = scaled_tile_w
        self.scaled_tile_h = scaled_tile_h
        #self.tissue_percentage = tissue_percentage
        self.tile_score_thresh = tile_score_thresh
        self.level = level
        self.best_level_for_downsample = best_level_for_downsample
        self.real_scale_factor = real_scale_factor
        self.tiles = []
        self.rois = rois
        self.grid_manager = grid_manager

        
    def change_level_of_rois(self, new_level:int):
        """
            convenience function to change the level for all rois at once in place
        """
        for roi in self.rois:
            if roi != None:
                roi.change_level_in_place(new_level)

    #def __str__(self):
    #    return summary_title(self) + "\n" + summary_stats(self)

    #def mask_percentage(self):
    #    """
    #    Obtain the percentage of the slide that is masked.
#
    #    Returns:
    #       The amount of the slide that is masked as a percentage.
    #    """
    #    return 100 - self.tissue_percentage

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
        Retrieve the tiles ranked by score. If rois were specified, 
        only tiles within those rois will be taken into account.

        Returns:
           List of the tiles ranked by score.
        """
        sorted_list = sorted(self.tiles, key=lambda t: t.score, reverse=True)
        return sorted_list
    
    def top_tiles(self, verbose=False):
        """
        Retrieve only the tiles that pass scoring

        Returns:
           List of the top-scoring tiles.
        """
        sorted_tiles = self.tiles_by_score()
        top_tiles = [tile for tile in sorted_tiles
                     if self.check_tile(tile)]
        if verbose:
            print(f'{self.wsi_path}: Number of tiles that will be kept/all possible tiles: \
            {len(top_tiles)}/{len(sorted_tiles)}')
        return top_tiles

    def check_tile(self, tile):
        return tile.score > self.tile_score_thresh
            
    def show_wsi(self, 
                 figsize:tuple=(10,10), 
                 axis_off:bool=False):
        """
        Displays a scaled down overview image of the wsi.
        
        Arguments:
            figsize: Size of the plotted matplotlib figure containing the image.
            axis_off: bool value that indicates, if axis shall be plotted with the picture
        """
        wsi_pil, large_w, large_h, new_w, new_h, best_level_for_downsample = \
        wsi_to_scaled_pil_image(wsi_filepath=self.wsi_path, 
                                scale_factor=self.scale_factor, 
                                level=0)  
        
        # Create figure and axes
        fig,ax = plt.subplots(1,1,figsize=figsize)    
        # Display the image
        ax.imshow(wsi_pil)     
        if(axis_off):
            ax.axis('off') 
        plt.show() 
    
    
    def show_wsi_with_all_possible_tiles(self, 
                                   figsize:Tuple[int] = (10,10),
                                   scale_factor:int = 32, 
                                   axis_off:bool = False):
        """    
        Loads a whole slide image, scales it down, converts it into a numpy array and displays 
        it with a grid overlay for all tiles
        that could fit in the given rois or if no rois are given in the whole wsi.
        Arguments:
            figsize: Size of the plotted matplotlib figure containing the image.
            scale_factor: The larger, the faster this method works, 
                            but the plotted image has less resolution.
            axis_off: bool value that indicates, if axis shall be plotted with the picture
        """
        self.grid_manager.show_wsi_with_rois_and_grids(figsize=figsize, 
                                                       scale_factor=scale_factor, 
                                                       axis_off=axis_off)

    def show_wsi_with_top_tiles(self, 
                                   figsize:Tuple[int] = (10,10),
                                   scale_factor:int = 32, 
                                   axis_off:bool = False):
        """    
        Loads a whole slide image, scales it down, converts it into a numpy array and displays 
        it with a grid overlay for all tiles
        that passed scoring to visualize which tiles 
        e.g. "tiles.WsiIToTilesParallel" calculated as worthy to keep.
        Arguments:
            figsize: Size of the plotted matplotlib figure containing the image.
            scale_factor: The larger, the faster this method works, 
                            but the plotted image has less resolution.
            axis_off: bool value that indicates, if axis shall be plotted with the picture
        """
        util.show_wsi_with_rois(wsi_path=self.wsi_path, 
                        rois=self.rois+[t.rectangle.as_roi(level=t.level) for t in self.top_tiles()], 
                        figsize=figsize, 
                        scale_factor=scale_factor, 
                        axis_off=axis_off)
        
    def show_wsi_with_rois(self, 
                           figsize:Tuple[int] = (10,10),
                           scale_factor:int = 32, 
                           axis_off:bool = False):
        """    
        Loads a whole slide image, scales it down, 
        converts it into a numpy array and displays it with a grid overlay for all rois
        specified in self.wsi_info.rois
        Arguments:
            figsize: Size of the plotted matplotlib figure containing the image.
            scale_factor: The larger, the faster this method works, 
            but the plotted image has less resolution.
            axis_off: bool value that indicates, if axis shall be plotted with the picture
        """
        show_wsi_with_rois(self.wsi_path, self.rois, axis_off=axis_off)
        
        
############################# functions #########################################

###
# some example/default implementations for functions that other methods below take as arguments
###

def score_tile_by_tp_and_cf(tile_pil:PIL.Image.Image, scoring_function:Callable)->Tuple[float, Dict]:
    """
    Arguments:
        tile_pil: The tile as a PIL Image
        a function that takes the tissue percentage and the combined factor as arguments
    Return:
        The tile's score and a dictionary with all factors that were calculated from the PIL Image and 
        used for calculating the score.
    """
    tile_pil_filtered = filter.filter_img(tile_pil)
    if(tile_pil_filtered is None):
        return 0.0, None
    
    tile_np = pil_to_np_rgb(tile_pil)
    tile_np_filtered = pil_to_np_rgb(tile_pil_filtered)
    tissue_percentage = filter.tissue_percent(tile_np_filtered)
    
    color_factor = hsv_purple_pink_factor(tile_np)
    s_and_v_factor = hsv_saturation_and_value_factor(tile_np)
    
    combined_factor = color_factor * s_and_v_factor   
    score = scoring_function(tissue_percentage, combined_factor)
                    
    # scale score to between 0 and 1
    score = 1.0 - (10.0 / (10.0 + score))
                  
    return score, {"color_factor":color_factor, "s_and_v_factor":s_and_v_factor}

def score_tile_1(tile_pil:PIL.Image.Image)->Tuple[float, Dict]:
    """
    Arguments:
        tile_pil: The tile as a PIL Image
    Return:
        The tile's score and a dictionary with all factors that were calculated from the PIL Image and 
        used for calculating the score.
    """
    return score_tile_by_tp_and_cf(tile_pil=tile_pil, scoring_function=scoring_function_1)

def score_tile_2(tile_pil:PIL.Image.Image)->Tuple[float, Dict]:
    """
    Arguments:
        tile_pil: The tile as a PIL Image
    Return:
        The tile's score and a dictionary with all factors that were calculated from the PIL Image and 
        used for calculating the score.
    """
    return score_tile_by_tp_and_cf(tile_pil=tile_pil, scoring_function=scoring_function_2)

def __scoring_function_muscle(tissue_percent, combined_factor):
    """
    This favors pink over purple regions (muscle tissue has few nuclei and lots of sarcoplam, which
    is pink in H&E staining). color_factor and s_and_v_factor are higher, 
    the more hematoxylin stained tissue is in the image
    """
    if(combined_factor == 0):
        return tissue_percent / 2
    else:
        return (100/(combined_factor)) * tissue_percent / 100   

    
def scoring_function_muscle(tile_pil:PIL.Image.Image)->Tuple[float, Dict]:
    """
    Arguments:
        tile_pil: The tile as a PIL Image
    Return:
        The tile's score and a dictionary with all factors that were calculated from the PIL Image and 
        used for calculating the score.
    """
    tile_pil_filtered = filter.filter_img(tile_pil)
    if(tile_pil_filtered is None):
        return 0.0, None
    
    tile_np = pil_to_np_rgb(tile_pil)
    tile_np_filtered = pil_to_np_rgb(tile_pil_filtered)
    tissue_percentage = filter.tissue_percent(tile_np_filtered)
    
    color_factor = hsv_purple_pink_factor(tile_np)
    s_and_v_factor = hsv_saturation_and_value_factor(tile_np)
    
    combined_factor = color_factor * s_and_v_factor   
    score = __scoring_function_muscle(tissue_percentage, combined_factor)
                    
    # scale score to between 0 and 1
    score = 1.0 - (10.0 / (10.0 + score))
                  
    return score, {"color_factor":color_factor, "s_and_v_factor":s_and_v_factor}


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
        x: x-coordinate of the upper left pixel. The method assumes, 
            that you know the dimensions of your specified level.
        y: y-coordinate of the upper left pixel. The method assumes, 
            that you know the dimensions of your specified level.
        width: tile width
        height: tile height
        level: Level of the WSI you want to extract the tile from. 
                0 means highest resolution.
        
    Return:
        tile as PIL.Image as RGB
    """
    s = slide.open_slide(str(path))
    tile_region = s.read_region((int(x), int(y)), level, (int(width), int(height)))
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
                    grids_per_roi:int,
               tiles_folder_path:pathlib.Path,
               tile_height:int, 
               tile_width:int,
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
               grids_per_roi:int,
               tile_height:int, 
               tile_width:int,
               tile_naming_func:Callable = tile_naming_function_default,
               tile_score_thresh:float = 0.55,
               tile_scoring_function = score_tile_1,
               optimize_grid_angles:bool = False,
               angle_stepsize:float = 5,
               level = 0, 
               save_tiles:bool = False,
               tiles_folder_path:pathlib.Path = None,
               rois:List[shared.roi.RegionOfInterestPolygon] = None,
               minimal_tile_roi_intersection_ratio:float = 1.0,     
               verbose=False)-> Union[TileSummary, pandas.DataFrame]:
    """
    Calculates tile coordinates and returns a TileSummary object. 
    If save_tiles == True the tiles will also be extracted
    and saved from the WSI or ROI (ROI is assumed to be a "normal" image format like .png).
    
    Arguments:
    wsi_path: Path to a WSI or ROI(=already extracted part of a wsi in e.g. .png format)
    grids_per_roi: Use multiple grids per roi to enhance the number of tiles.
                            The grids will be shifted depending on the number of grids 
                            and the tile_width, tile_height.
                            e.g.: grids_per_roi == 3 and tile_width == tile_height == 1024
                                  First grid starts at (0,0).
                                  Second grid starts at (1024*1/3 , 1024*1/3)
                                  Third grid starts at (1024*2/3 , 1024*2/3)
    tile_heigth: Number of pixels tile height.
    tile_width: Number of pixels tile width.
    
    tile_score_thresh: Tiles with a score higher than the number from "tileScoringFunction" will be saved.
    tile_scoring_function: Function to score one tile to determine if it should be saved or not.
    optimize_grid_angles: Finds the best rotation of the tiles for each roi to fit in
                            the most tiles
    angle_stepsize: only relevant if optimize_grid_angles is True. The process of finding the best
                    angle is iterative. The smaller the stepsize, the closer is the result
                    to the best angle, but the longer the computation takes.
    tile_naming_func: 99% of the time there should be no necessity to change this.
                        A function, that takes a pathlib.Path to the WSI or ROI as an argument 
                        and returns a string.
                        This string will then be used as part of the name for the tile 
                        (plus some specific tile information and
                        the file format .png, whick is generated by this library).
                        
    level: Level of the WSI you want to extract the tiles from. 0 means highest resolution. 
            For not wsi formats like .png leave it at 0.
    save_tiles: if True the tiles will be extracted and saved to {tilesFolderPath}
    tiles_folder_path: The folder where the extracted tiles will be saved (only needed if save_tiles=True).
    return_as_tilesummary_object: return_as_tilesummary_object: Set this to true, if you 
                                    want the TileSummary object and not a pandas dataframe.
    rois: If rois are specified, only tissue inside these rois will be extracted.
          Rois are deepcopied and overlapping rois of these deepcopies are merged.
    minimal_tile_roi_intersection_ratio: (0.0, 1.0] 
                        (intersection area between roi and tile)/tile area >= tile_roi_intersection_ratio so 
                        that a tile will be used for further calculations
    Return:
       a TileSummary object
    """    

    if(tiles_folder_path is None and save_tiles == True):
        raise ValueError("You should specify a {tiles_folder_path}")
    
    if verbose:
        print(f"Starting to process {str(wsi_path)}")

    scale_factor = 32

    ### against DecompressionBombWarning
    #mage.MAX_IMAGE_PIXELS = 10000000000000
    openslide.lowlevel._load_image = openslide_overwrite._load_image
    
    
    wsi = openslide.open_slide(str(wsi_path))
    wsi_original_width, wsi_original_height = wsi.level_dimensions[level]
    best_level_for_downsample = wsi.get_best_level_for_downsample(scale_factor)
    wsi_scaled_width, wsi_scaled_height = wsi.level_dimensions[best_level_for_downsample]
    
    #print(wsi_path)
    #print(tile_width)
    #print(tile_height)
    #print()
    #print(grids_per_roi)
    #print(level)
    
    gm = GridManager(wsi_path=wsi_path, 
                    tile_width=tile_width, 
                     tile_height=tile_height, 
                     rois=rois, 
                     grids_per_roi=grids_per_roi, 
                     level=level)
    if(optimize_grid_angles):
        gm.optimize_grid_angles(stepsize=angle_stepsize, 
                                minimal_intersection_quotient=minimal_tile_roi_intersection_ratio)
    else:
        gm.filter_grids(minimal_intersection_quotient=minimal_tile_roi_intersection_ratio)
    
    
    print(len(gm.get_all_rectangles()))
    
    
    
    real_scale_factor = int(math.pow(2,best_level_for_downsample-level))
    tile_height_scaled = adjust_level(tile_height, level, best_level_for_downsample)
    tile_width_scaled = adjust_level(tile_width, level, best_level_for_downsample)
    
    tilesummary = TileSummary(wsi_path=wsi_path,
                           tiles_folder_path=tiles_folder_path,
                             orig_w=wsi_original_width,
                             orig_h=wsi_original_height,
                             orig_tile_w=tile_width,
                             orig_tile_h=tile_height,
                             scale_factor=scale_factor,
                             scaled_w=wsi_scaled_width,
                             scaled_h=wsi_scaled_height,
                             scaled_tile_w=tile_width_scaled,
                             scaled_tile_h=tile_height_scaled,
                             tile_score_thresh=tile_score_thresh,
                             level=level,
                             best_level_for_downsample=best_level_for_downsample,
                             real_scale_factor=real_scale_factor,
                             rois=gm.rois, #rois in gm are deepcopied and overlapping rois are merged 
                             grid_manager=gm)  
    
    wh = WsiHandler(wsi_path=wsi_path)
    
    for n, r in enumerate(gm.get_all_rectangles()):        
        rect_tile_downsampled = r.deepcopy().change_level(current_level=level, 
                                                     new_level=best_level_for_downsample)       
        tile_pil_scaled_down = wh.extract_tile_from_wsi_2(rectangle_tile=rect_tile_downsampled, 
                                                          level=best_level_for_downsample)
        
        #factors_dict is a dictionary which contains all factors 
        #that were relevant for determining the score
        #inside of the tile_scoring_function
        score, factors_dict = tile_scoring_function(tile_pil=tile_pil_scaled_down)  
        
        try:
            tile = Tile(tilesummary=tilesummary,
                             tiles_folder_path=tiles_folder_path, 
                             tile_num = n, 
                             rectangle = r,
                             rectangle_downsampled = rect_tile_downsampled, 
                             score=score,
                             dict_with_all_parameters_to_determine_score=factors_dict,
                             tile_naming_func=tile_naming_func, 
                             level=level, 
                             level_downsampled=best_level_for_downsample, 
                             real_scale_factor=real_scale_factor, 
                            grid_manager=gm,
                             roi=gm.get_rois_the_given_tile_is_in(rect_tile=rect_tile_downsampled,
                                                                  rect_level=best_level_for_downsample,
                                            minimal_tile_roi_intersection_ratio=minimal_tile_roi_intersection_ratio)[0])
            tilesummary.tiles.append(tile)
        except IndexError as e:
            print(f'IndexError in WsiToTiles for the rectangle: {r}')
    
  
    if(save_tiles):
        for tile in tilesummary.top_tiles(verbose):
            tile.save_tile()
            
    if verbose:
        tilesummary.top_tiles(verbose)
        
    return tilesummary

def WsiOrROIToTilesMultithreaded(wsi_paths:List[pathlib.Path],
                             grids_per_roi:int,
                             tiles_folder_path:pathlib.Path,
                             tile_height:int, 
                             tile_width:int,
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
                             grids_per_roi:int,
                             tile_height:int, 
                             tile_width:int,
                             tile_naming_func:Callable = tile_naming_function_default,
                             tile_score_thresh:float = 0.55,
                             tile_scoring_function = score_tile_1,
                             optimize_grid_angles:bool = False,
                             level = 0, 
                             save_tiles:bool = False,
                             tiles_folder_path:pathlib.Path=None,
                             wsi_path_to_rois:Dict[pathlib.Path, List[shared.roi.RegionOfInterestPolygon]] = None,
                             minimal_tile_roi_intersection_ratio:float = 1.0,
                             verbose=False, 
                             number_of_processes = None)->List[TileSummary]:
    """
    The method WsiToTiles for a list of WSIs/ROIs in parallel.
    
    Arguments:
    wsi_paths: A list of paths to the WSIs or ROIs(=preextracted png files from WSIs)
    grids_per_roi: Use multiple grids per roi to enhance the number of tiles.
                            The grids will be shifted depending on the number of grids 
                            and the tile_width, tile_height.
                            e.g.: grids_per_roi == 3 and tile_width == tile_height == 1024
                                  First grid starts at (0,0).
                                  Second grid starts at (1024*1/3 , 1024*1/3)
                                  Third grid starts at (1024*2/3 , 1024*2/3)
    tile_heigth: Number of pixels tile height.
    tile_width: Number of pixels tile width.
                                    
    tile_score_thresh: Tiles with a score higher than the number from "tileScoringFunction" will be saved.
    tile_scoring_function: Function to score one tile to determine if it should be saved or not.
    optimize_grid_angles: Finds the best rotation of the tiles for each roi to fit in
                            the most tiles
    tile_naming_func: 99% of the time there should be no necessity to change this.
                        A function, that takes a pathlib.Path to the WSI 
                        or ROI as an argument and returns a string.
                        This string will then be used as part of the name 
                        for the tile (plus some specific tile information and
                        the file format .png, whick is generated by this library).
    level: Level of the WSI you want to extract the tile from. 0 means highest resolution. 
            For not wsi formats like .png leave it at 0.
    tiles_folder_path: The folder where the extracted tiles will be saved (only needed if save_tiles=True).
    return_as_tilesummary_object: Set this to true, 
                                    if you want the TileSummary object 
                                    and not a pandas dataframe.
    wsi_path_to_rois: a dict with key: wsi_path and value List[shared.roi.RegionOfInterestPolygon]].
                        Rois are deepcopied and overlapping rois of these deepcopies are merged.
    minimal_tile_roi_intersection_ratio: (0.0, 1.0] 
                        (intersection area between roi and tile)/tile area >= tile_roi_intersection_ratio so 
                        that a tile will be used for further calculations
    number_of_processes: if None, number_of_processes will be the number of cpu cores
    
    Return:
       a List of TileSummary objects

    """
    
    pbar = tqdm(total=len(wsi_paths))
    tilesummaries = []
    def update(ts):
        tilesummaries.append(ts)
        pbar.update()
        
    def error(e):
        print(e)
        
    with multiprocessing.Pool(processes=number_of_processes) as pool:
        for p in wsi_paths:
            pool.apply_async(WsiToTiles, 
                             kwds={"wsi_path":p,
                                   "grids_per_roi":grids_per_roi,
                                   "tile_height":tile_height, 
                                   "tile_width":tile_width,
                                   "tile_naming_func":tile_naming_func,
                                   "tile_score_thresh":tile_score_thresh, 
                                   "tile_scoring_function":tile_scoring_function,
                                   "optimize_grid_angles":optimize_grid_angles,
                                   "level":level, 
                                   "save_tiles":save_tiles,
                                   "tiles_folder_path":tiles_folder_path,
                                   "rois": util.safe_dict_access(wsi_path_to_rois, p),
                                   "minimal_tile_roi_intersection_ratio":minimal_tile_roi_intersection_ratio,
                                   "verbose":verbose}, 
                                   callback=update, 
                                   error_callback=error)
                                    
        pool.close()
        pool.join()
    
    return tilesummaries
       
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

def tile_to_pil_tile(tile:Tile):
      """
      Convert tile information into the corresponding tile as a PIL image read from the whole-slide image file.

      Args:
        tile: Tile object.

      Return:
        Tile as a PIL image.
      """
      return ExtractTileFromWSI(tile.tilesummary.wsi_path, 
                                tile.get_x(), 
                                tile.get_y(), 
                                tile.get_width(), 
                                tile.get_height(), 
                                tile.level)


def tile_to_np_tile(tile):
  """
  Convert tile information into the corresponding tile 
  as a NumPy image read from the whole-slide image file.

  Args:
    tile: Tile object.

  Return:
    Tile as a NumPy image.
  """
  pil_img = tile_to_pil_tile(tile)
  np_img = pil_to_np_rgb(pil_img)
  return np_img



def get_tile_image_path(tile:Tile):
  """
  Obtain tile image path based on tile information such 
  as row, column, row pixel position, column pixel position,
  pixel width, and pixel height.

  Args:
    tile: Tile object.

  Returns:
    Path to image tile.
  """
  t = tile
  if tile.tiles_folder_path is None:
      return os.path.join(tile.tile_naming_func(tile.tile_summary.wsi_path) + "-" + 'tile'\
                          + "-r%d-c%d-x%d-y%d-w%d-h%d" % (
                        t.r, t.c, t.o_c_s, t.o_r_s, t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s) + "." + 'png')
  else:
      return os.path.join(tile.tiles_folder_path, 
                          tile.tile_naming_func(tile.tile_summary.wsi_path) + "-" + 'tile'\
                          + "-r%d-c%d-x%d-y%d-w%d-h%d" % (
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
  Function to reduce scores of tiles with narrow HSV saturations and values since saturation 
  and value standard
  deviations should be relatively broad if the tile contains significant tissue.

  Example of a blurred tile that should not be ranked as a top tile:
    ../data/tiles_png/006/TUPAC-TR-006-tile-r58-c3-x2048-y58369-w1024-h1024.png

  Args:
    rgb: RGB image as a NumPy array

  Returns:
    Saturation and value factor, where 1 is no effect 
    and less than 1 means the standard deviations of saturation and
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
  Compute scoring factor based on purple and pink HSV hue deviations 
  and degree to which a narrowed hue color range
  average is purple versus pink.

  Args:
    rgb: Image an NumPy array.

  Returns:
    Factor that favors purple (hematoxylin stained) 
    tissue over pink (eosin stained) tissue.
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
  Function to favor purple (hematoxylin) over pink (eosin) 
  staining based on the distance of the HSV hue average
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


###################################### more imports ################################################################

from shared.tile import Tile