from __future__ import annotations #https://stackoverflow.com/questions/33837918/type-hints-solve-circular-dependency

from typing import List, Tuple, Union, Sequence
from shared.wsi import WholeSlideImage
from shared.case import Case
from tile_extraction import util
from util import polygon_to_numpy
import copy
import shapely
from shapely.geometry import Polygon, Point
import numpy as np
import json

from abc import ABC, abstractmethod


class RegionOfInterest(ABC):
    __removed = False #Flag that can be set True, to mark it as "deleted" for the patient_manager. use getter and setter methods
                      # this flag is not used in the TileSummary class
    
    roi_id:str = None
    whole_slide_image:WholeSlideImage = None
    labels:List[int] = None
    __tiles:List[Tile] = None
    def __init__(self, roi_id:str, whole_slide_image:WholeSlideImage = None):
        self.roi_id = roi_id
        self.whole_slide_image = whole_slide_image
        self.__tiles = []
        
    def is_removed(self):
        return self.__removed
    
    def set_removed_flag(self, value:bool):
        self.__removed = value
        for tile in self.get_tiles():
            tile.set_removed_flag(value)
            
    def get_tiles(self):
        return [t for t in self.__tiles if(not t.is_removed())]
    
    def add_tile(self, tile:Tile):
        self.__tiles.append(tile)
        
    def reset_tiles(self):
        self.__tiles = []
        
        
class RegionOfInterestDummy(RegionOfInterest):
    """
    Used when there is no necessity for a roi. E.g. when there are already preextracted tiles
    Just here to persist the hierarchy of the classes
    """
    __tiles:List[pathlib.Path]
    def __init__(self, roi_id:str, whole_slide_image:WholeSlideImage):
        super().__init__(roi_id = roi_id, whole_slide_image = whole_slide_image)
        self.__tiles = []
        
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
        super().__init__(roi_id=roi_id)
        self.polygon = Polygon(vertices)
        self.level = level
        
        
    def __repr__(self):
        return f"Vertices coordinates: {self.get_vertices()}, level: {self.level}"
    
    def get_vertices(self)->np.ndarry:
        """
        Returns a numpy array with shape [number_of_vertices, 2] where the second dimension represents x,y-coordinates of
        each vertex
        """       
        return polygon_to_numpy(self.polygon)
       
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
    
class __PolygonHelper:
    def __init__(self, level:int, vertices:Sequence[Tuple[float, float]]):
        self.level = level
        self.vertices = vertices

def __get_polygons_from_json(json_path:pathlib.Path)->List[__PolygonHelper]:
    """
    Reads the json file and returns a list of __PolygonHelper objects. 
    This should be a specialized function for the specific structure of your json files.
    
    Arguments:
        json_path: path to json file
        
    Returns:
        List of __PolygonHelper objects
    """
    polygons = []
    with open(json_path) as json_file:
        for annotatin in json.load(json_file):             
            if(annotatin["geometry"]["type"] == 'MultiPolygon'):
                multi_polygon_vertices = annotatin["geometry"]["coordinates"]
                #print(f'Multi Polygon: {np.array(multi_polygon_vertices).squeeze().shape}')
                ##QuPath produces Polygons and Multipolygons 
                ##(see difference here: https://gis.stackexchange.com/questions/225368/understanding-difference-between-polygon-and-
                ##multipolygon-for-shapefiles-in-qgis/225373)
                ##This loop separates Multipolygons into individual Polygons
                for sub_polygon_vertices in multi_polygon_vertices:
                    sub_polygon_vertices_array = np.array(sub_polygon_vertices).squeeze()
                    if(len(sub_polygon_vertices_array.shape) == 2 and sub_polygon_vertices_array.shape[1] == 2):
                        #print(f'then: {sub_polygon_vertices_array.shape}')
                        polygons.append(__PolygonHelper(level=0, vertices=sub_polygon_vertices_array))
                    else:
                        for elem in sub_polygon_vertices_array:
                            elem_array = np.array(elem).squeeze()
                            #print(f'else: {elem_array.shape}')
                            polygons.append(__PolygonHelper(level=0, vertices=elem_array))
                
            elif(annotatin["geometry"]["type"] == 'Polygon'):
                vertices = annotatin["geometry"]["coordinates"]
                #print(f'Polygon: {np.array(vertices).squeeze().shape}')
                polygons.append(__PolygonHelper(level=0, vertices=np.array(vertices).squeeze()))
            else:
                assert False
            
            #coords_raw = polygon_coords["geometry"]["coordinates"]

            #for vertices in coords_raw:
            #    #all polygons on level 0 by default in this case                
            #    polygons.append(__PolygonHelper(level=0, vertices=np.array(vertices).squeeze()))
    return polygons

def __validate_polygons(polygons:Sequence[__PolygonHelper])->Sequence[__PolygonHelper]:
    """
    Validates and if necessary fixes issues of the vertices of a sequence of polygons.
    Migth even delete a "polygon" if it consists of less than three vertices.
    E.g. solves self-intersections etc.
      
    Arguments:
        polygons: A sequence of __PolygonHelper objects.
    Returns:
        A new validated sequence of __PolygonHelper objects.
    """
    
    validated_polygons = []
    for polygon in polygons:
        #remove structures with less than three vertices
        if(len(polygon.vertices) >= 3):
            #TODO: wite some more validation steps here if necessary
            validated_polygons.append(polygon)
            
    return validated_polygons
    
    
def get_list_of_RegionOfInterestPolygon_from_json(json_path:pathlib.Path, 
                                                  polygons_from_json_func:Callable=__get_polygons_from_json, 
                                                  validate_polygons_func:Callable=__validate_polygons)\
                                                    ->List[RegionOfInterestPolygon]:
    """
    Arguments:
        json_path: path to json file
        polygons_from_json_func: A function that reads the json file and returns a list of __PolygonHelper objects.
        validate_polygons_func: A function that validates if necessary fixes a list of __PolygonHelper objects.
        
    """
    rois = []
    for n, polygon_helper in enumerate(validate_polygons_func(polygons_from_json_func(json_path))):
        roi_id = f'{json_path.stem}_roi_number_{str(n)}'
        rois.append(RegionOfInterestPolygon(roi_id=roi_id, vertices=polygon_helper.vertices, level=polygon_helper.level))
    return rois


def merge_overlapping_rois(rois:List[RegionOfInterestPolygon]):
    """
    merges overlapping rois
    """
    if(rois is None):
        raise ValueError('rois must not be None')
    #for r in rois:
    #    if(type(r) is not RegionOfInterestPolygon
    #       or type(r) is not type(wsi_processing_pipeline.shared.roi.RegionOfInterestPolygon)):
    #        print(type(r))
    #        raise ValueError('the rois must be of the type RegionOfInterestPolygon')
    
    if(len(rois) > 1):
        roi_0 = rois[0]
        intersecting_rois = []
        for r in rois[1:]:
            if(roi_0.polygon.intersection(r.polygon).area > 0):
                    intersecting_rois.append(r)
        if(len(intersecting_rois) == 0):
            return
            
        for r in [roi_0] + intersecting_rois:
            rois.remove(r)
            
                
        merged_poly = roi_0.polygon
        merged_roi_id = roi_0.roi_id
        for r in intersecting_rois:
            merged_poly = merged_poly.union(r.polygon)
            merged_roi_id = merged_roi_id + " + " + r.roi_id
        merged_roi = RegionOfInterestPolygon(roi_id=merged_roi_id, 
                                             vertices=polygon_to_numpy(polygon=merged_poly),
                                             level=roi_0.level)
        rois.append(merged_roi)
        merge_overlapping_rois(rois=rois)