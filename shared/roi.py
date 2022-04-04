from __future__ import annotations #https://stackoverflow.com/questions/33837918/type-hints-solve-circular-dependency

from typing import List, Tuple, Union, Sequence
from shared.wsi import WholeSlideImage
from shared.case import Case
from tile_extraction import util, tiles
from util import polygon_to_numpy
import copy
import shapely
from shapely.geometry import Polygon, Point
import numpy as np
import json

from abc import ABC, abstractmethod
from enum import Enum
import PIL
import pickle


class RegionOfInterest(ABC):
    __removed = False #Flag that can be set True, to mark it as "deleted" for the patient_manager. use getter and setter methods
                      # this flag is not used in the TileSummary class
    
    roi_id:str = None
    whole_slide_image:WholeSlideImage = None
    labels:List[Union[int,str]] = None
    __tiles:List[Tile] = None
    def __init__(self, roi_id:str, whole_slide_image:WholeSlideImage = None, labels:List[Union[int,str]]=None):
        self.roi_id = roi_id
        self.whole_slide_image = whole_slide_image
        self.labels = labels
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
                 level:int, 
                 labels:List[Union[int,str]]=None):
        """
            Arguments:
            roi_id: a unique id for the roi
            vertices: numpy array of the shape [number_of_vertices, 2] ( == x-,y-coordinate)
            level: level of the whole-slide image. 0 means highest resolution. 
                    Leave it 0 if you use e.g. png files instead of a 
                    whole-slide image format like .ndpi
            labels: List of classification labels
        """
        super().__init__(roi_id=roi_id, labels=labels)
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
    def __init__(self, level:int, vertices:Sequence[Tuple[float, float]], labels:List[Union[int,str]]=None):
        self.level = level
        self.vertices = vertices
        self.labels = labels

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
        for annotation in json.load(json_file):             
            if(annotation["geometry"]["type"] == 'MultiPolygon'):
                multi_polygon_vertices = annotation["geometry"]["coordinates"]
                #print(f'Multi Polygon: {np.array(multi_polygon_vertices).squeeze().shape}')
                ##QuPath produces Polygons and Multipolygons 
                ##(see difference here: https://gis.stackexchange.com/questions/225368/understanding-difference-between-polygon-and-
                ##multipolygon-for-shapefiles-in-qgis/225373)
                ##This loop separates Multipolygons into individual Polygons
                for sub_polygon_vertices in multi_polygon_vertices:
                    sub_polygon_vertices_array = np.array(sub_polygon_vertices, dtype=object).squeeze()
                    if(len(sub_polygon_vertices_array.shape) == 2 and sub_polygon_vertices_array.shape[1] == 2):
                        #print(f'then: {sub_polygon_vertices_array.shape}')
                        polygons.append(__PolygonHelper(level=0, vertices=sub_polygon_vertices_array))
                    else:
                        for elem in sub_polygon_vertices_array:
                            elem_array = np.array(elem).squeeze()
                            #print(f'else: {elem_array.shape}')
                            polygons.append(__PolygonHelper(level=0, vertices=elem_array))
                
            elif(annotation["geometry"]["type"] == 'Polygon'):
                vertices = annotation["geometry"]["coordinates"]
                #print(f'Polygon: {np.array(vertices).squeeze().shape}')
                polygons.append(__PolygonHelper(level=0, vertices=np.array(vertices, dtype=object).squeeze()))
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
        rois.append(RegionOfInterestPolygon(roi_id=roi_id, 
                                            vertices=polygon_helper.vertices, 
                                            level=polygon_helper.level, 
                                            labels = polygon_helper.labels))
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
            
            try:
                if(roi_0.polygon.intersection(r.polygon).area > 0):
                    intersecting_rois.append(r)
            except shapely.geos.TopologicalError as e:
                 #possible temporary fix could be "roi.polygon.buffer(0).intersection(rect_as_roi.polygon).area"
                 #as described here: https://github.com/gboeing/osmnx/issues/278
                 #but buffer(0) migth change the roi in an unexpectec way
                 #intersection_area = roi.polygon.buffer(0).intersection(rect_as_roi.polygon).area
                 
                 #print(f'method tiles.Grid.filter_grid: {e}')
                 continue
           
        if(len(intersecting_rois) == 0):
            return
            
        for r in [roi_0] + intersecting_rois:
            rois.remove(r)
            
                
        merged_poly = roi_0.polygon
        merged_roi_id = roi_0.roi_id
        merged_labels = []
        for r in intersecting_rois:
            merged_poly = merged_poly.union(r.polygon)
            merged_roi_id = merged_roi_id + " + " + r.roi_id
            merged_labels += r.labels
        merged_roi = RegionOfInterestPolygon(roi_id=merged_roi_id, 
                                             vertices=polygon_to_numpy(polygon=merged_poly),
                                             level=roi_0.level, 
                                             labels=merged_labels)
        rois.append(merged_roi)
        merge_overlapping_rois(rois=rois)
        
        
########
# roi adjustment for .mrxs wsi files, since openslide adds black padding
########


def is_row_black(img:Union[PIL.Image.Image, numpy.ndarray], row_num:int)->bool:
    if(type(img) is PIL.Image.Image):
        img = util.pil_to_np_rgb(img)
    return np.all(img[row_num, :, :] == [0,0,0])

def is_column_black(img:Union[PIL.Image.Image, numpy.ndarray], column_num:int)->bool:
    if(type(img) is PIL.Image.Image):
        img = util.pil_to_np_rgb(img)
    return np.all(img[:, column_num, :] == [0,0,0])

def get_num_of_black_rows_at_top(img:Union[PIL.Image.Image, numpy.ndarray])->int:
    n_rows = 0
    while(is_row_black(img=img, row_num=n_rows)):
        n_rows += 1
    return n_rows

def get_num_of_black_columns_at_left(img:Union[PIL.Image.Image, numpy.ndarray])->int:
    n_columns = 0
    while(is_column_black(img=img, column_num=n_columns)):
        n_columns += 1
    return n_columns

class Row_or_col(Enum):
    ROW = 1
    COLUMN = 2
    
def is_only_black_white(img:Union[PIL.Image.Image, numpy.ndarray], index:int, row_or_col:Row_or_col)->bool:
    """
    checks if the by index specified row or column in the image contains only [0,0,0] == black
    or [255,255,255] == white RGB values
    """
    if(type(img) is PIL.Image.Image):
        img = util.pil_to_np_rgb(img)
    a = None
    if(row_or_col is Row_or_col.ROW):
        a = img[index, :, :]
    elif(row_or_col is Row_or_col.COLUMN):
        a = img[:, index, :]
    else:
        raise ValueError('row_or_col has insufficient value')
    return np.where(np.logical_and(np.ravel(a) > 0, np.ravel(a) < 255))[0].size == 0

def get_num_of_only_black_white(img:Union[PIL.Image.Image, numpy.ndarray], row_or_col:Row_or_col)->int:
    n = 0
    while(is_only_black_white(img=img, index=n, row_or_col=row_or_col)):
        n += 1
    return n

def adjust_rois_mrxs(wsi_path:pathlib.Path, 
                rois:List[RegionOfInterestPolygon])->List[RegionOfInterestPolygon]:
    """
    Openslide adds black padding when opening a .mrxs file.
    That leads to misaligned ROIs.
    This method adjusts the coordinates of the ROIS.
    It returns new RegionOfInterestPolygon objects.
    """
    wsi_img_level = 5
    
    wh = tiles.WsiHandler(wsi_path=wsi_path)
    wsi_img = wh.get_wsi_as_pil_image(level=wsi_img_level)
    
    cols_left = get_num_of_only_black_white(img=wsi_img, row_or_col=Row_or_col.COLUMN)
    rows_top = get_num_of_only_black_white(img=wsi_img, row_or_col=Row_or_col.ROW)
    rois_adjusted = []
    for r in rois:
        new_vertices = util.polygon_to_numpy(r.polygon)
        new_vertices += [util.adjust_level(value_to_adjust=cols_left, from_level=wsi_img_level, to_level=r.level), 
                         util.adjust_level(value_to_adjust=rows_top, from_level=wsi_img_level, to_level=r.level)]
        r_new = RegionOfInterestPolygon(roi_id=r.roi_id, vertices=new_vertices, level=r.level, labels=r.labels)
        rois_adjusted.append(r_new)

    return rois_adjusted

def save_as_pickle(obj:object, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def adjust_mrxs_and_save_rois(wsi_path:pathlib.Path, 
                         rois:List[RegionOfInterestPolygon], 
                         save_directory:pathlib.Path)->List[RegionOfInterestPolygon]:        
    save_name = f'{wsi_path.stem}-rois_adjusted.pickle'
    save_path = save_directory/save_name
    if(not save_path.exists()):    
        wh = tiles.WsiHandler(wsi_path=wsi_path)       
        rois_adjusted = adjust_rois_mrxs(wsi_path=wsi_path, rois=rois)               
        save_as_pickle(rois_adjusted, save_path)
    else:
        print(f'Already exists: {save_path}')
        rois_adjusted = load_pickle(save_path)
        
    return rois_adjusted