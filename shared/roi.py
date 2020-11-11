from __future__ import annotations #https://stackoverflow.com/questions/33837918/type-hints-solve-circular-dependency

from typing import List, Tuple, Union
from shared.wsi import WholeSlideImage
from shared.case import Case
from tile_extraction import util
import copy
import shapely
from shapely.geometry import Polygon, Point
import numpy as np

from abc import ABC, abstractmethod
class RegionOfInterest(ABC):
    roi_id:str = None
    whole_slide_image:WholeSlideImage = None
    labels:List[int] = None
    tiles:List[Tile] = None
    def __init__(self, roi_id:str, whole_slide_image:WholeSlideImage):
        self.roi_id = roi_id
        self.whole_slide_image = whole_slide_image
        self.tiles = []
        
        
class RegionOfInterestDummy(RegionOfInterest):
    """
    Used when there is no necessity for a roi. E.g. when there are already preextracted tiles
    Just here to persist the hierarchy of the classes
    """
    tiles:List[pathlib.Path]
    def __init__(self, roi_id:str, whole_slide_image:WholeSlideImage):
        super().__init__(roi_id = roi_id, whole_slide_image = whole_slide_image)
        self.tiles = []
        
class RegionOfInterestPreextracted(RegionOfInterest):    
    path:pathlib.Path = None
    def __init__(self, roi_id:str,  path:pathlib.Path, whole_slide_image:WholeSlideImage):
        super().__init__(roi_id = roi_id, whole_slide_image = whole_slide_image)
        self.path = path

class RegionOfInterestPolygon(RegionOfInterest):
    """
    represents a polygonal region of interest within a whole-slide image
    """
    level:int = None
    polygon:shapely.geometry.Polygon = None

    def __init__(self,
                 roi_id:str,
                 vertices:np.ndarray,  
                 level:int):
        """
            Arguments:
            roi_id: a unique id for the roi
            vertices: numpy array of the shape [number_of_vertices, 2] ( == x-,y-coordinate)
            level: level of the whole-slide image. 0 means highest resolution. 
                    Leave it 0 if you use e.g. png files instead of a 
                    whole-slide image format like .ndpi
        """
        self.polygon = Polygon(vertices)
        self.level = level
        
    def __repr__(self):
        return f"Vertices coordinates: {self.get_vertices()}, level: {self.level}"
    
    def get_vertices(self)->np.ndarry:
        """
        Returns a numpy array with shape [number_of_vertices, 2] where the second dimension represents x,y-coordinates of
        each vertex
        """       
        return util.polygon_to_numpy(self.polygon)
       
    def change_level_in_place(self, new_level:int)->RegionOfInterestPolygon:
        """
        adjusts all properties to new level in place and also returns itself
        """
        adjusted_vertices = [] 
        for v in self.get_vertices():
            adjusted_x = util.adjust_level(value_to_adjust=v[0], from_level=self.level, to_level=new_level)
            adjusted_y = util.adjust_level(value_to_adjust=v[1], from_level=self.level, to_level=new_level)
            adjusted_vertex = shapely.geometry.Point(adjusted_x, adjusted_y)
            adjusted_vertices.append(adjusted_vertex)
        self.polygon = Polygon(adjusted_vertices)
        self.level = new_level
        return self
    
    def change_level_deep_copy(self, new_level:int)->RegionOfInterestPolygon:
        """
        returns deep copy of itself with adjusted properties
        """
        dc = copy.deepcopy(self)
        dc.change_level_in_place(new_level=new_level)
        return dc
    
#class RegionOfInterestDefinedByCoordinates(RegionOfInterest):
#    """
#    represents a region of interest within a whole-slide image
#    """
#    x_upper_left:int = None
#    y_upper_left:int = None
#    height:int = None
#    width:int = None
#    level:int = None
#
#    def __init__(self,
#                 roi_id:str,
#                 x_upper_left:int, 
#                 y_upper_left:int, 
#                 height:int, 
#                 width:int, 
#                 level:int):
#        """
#            Arguments:
#            roi_id:
#            x_upper_left: x coordinate of roi's upper left point
#            y_upper_left: y coordinate of roi's upper left point
#            height: roi's height in pixel
#            width: roi's width in pixel
#            level: level of the whole-slide image. 0 means highest resolution. Leave it 0 if you use e.g. png files instead of a 
#                    whole-slide image format like .ndpi
#        """
#        super().__init__(roi_id = roi_id, whole_slide_image = None)
#        self.x_upper_left = x_upper_left
#        self.y_upper_left = y_upper_left
#        self.height = height
#        self.width = width
#        self.level = level
#        
#    def __repr__(self):
#        return f"x: {self.x_upper_left}, y: {self.y_upper_left}, height: {self.height}, width: {self.width}, level: {self.level}"
#        
#    def change_level_in_place(self, new_level:int):
#        """
#        adjusts all properties to new level in place and also returns itself
#        """
#        self.x_upper_left = util.adjust_level(value_to_adjust=self.x_upper_left, from_level=self.level, to_level=new_level)
#        self.y_upper_left = util.adjust_level(value_to_adjust=self.y_upper_left, from_level=self.level, to_level=new_level)
#        self.height = util.adjust_level(value_to_adjust=self.height, from_level=self.level, to_level=new_level)
#        self.width = util.adjust_level(value_to_adjust=self.width, from_level=self.level, to_level=new_level)
#        self.level = new_level
#        return self
#    
#    def change_level_deep_copy(self, new_level:int):
#        """
#        returns deep copy of itself with adjusted properties
#        """
#        dc = copy.deepcopy(self)
#        dc.x_upper_left = util.adjust_level(value_to_adjust=self.x_upper_left, from_level=self.level, to_level=new_level)
#        dc.y_upper_left = util.adjust_level(value_to_adjust=self.y_upper_left, from_level=self.level, to_level=new_level)
#        dc.height = util.adjust_level(value_to_adjust=self.height, from_level=self.level, to_level=new_level)
#        dc.width = util.adjust_level(value_to_adjust=self.width, from_level=self.level, to_level=new_level)
#        dc.level = new_level
#        return dc
#    