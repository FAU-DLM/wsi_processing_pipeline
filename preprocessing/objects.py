import wsi_processing_pipeline
from wsi_processing_pipeline.tile_extraction import tiles
from wsi_processing_pipeline.tile_extraction.tiles import DatasetType
import sklearn
import sklearn.model_selection
from enum import Enum
import typing
from typing import List, Callable
import pandas
import pandas as pd
import functools
from functools import partial
from sklearn.model_selection import StratifiedKFold, KFold
import pathlib
import multiprocessing
from tqdm import tqdm


class NamedObject():    
    def __init__(self,                 
                 path=None,
                 patient_id=None, 
                 case_id=None, 
                 slide_id=None,
                 classification_labels=None,
                 dataset_type:DatasetType=None, 
                 tiles:List[wsi_processing_pipeline.tile_extraction.tiles.Tile]=None):
        
        self.path=path
        self.patient_id=patient_id
        self.case_id=case_id
        self.slide_id=slide_id
        self.classification_labels=classification_labels
        self.dataset_type=dataset_type
        if isinstance(dataset_type, Enum):
            self.is_valid=True if dataset_type==DatasetType.validation else False 
        else:
            self.is_valid=None
        self.tiles = tiles
            
    def export_dataframe(self):
        df = pd.DataFrame(data={
        'path':self.path ,
        'patient_id':self.patient_id,
        'case_id':self.case_id,
        'slide_id':self.slide_id,
        'classification_labels':self.classification_labels,
        'dataset_type':self.dataset_type,
        'is_valid' : is_valid}, 
         index=[0])
        return df

    
##########################################
##########################################


def __create_named_object_by_slide_id(slide_id:str, 
                                    tile_paths:List[pathlib.Path], 
                                    patient_id_getter:Callable, 
                                    case_id_getter:Callable, 
                                    slide_id_getter:Callable, 
                                    classification_labels_getter:Callable)->NamedObject:
        #extract only those tile paths, that belong to the same slide
        paths = [p for p in tile_paths if slide_id == slide_id_getter(p)]
        #create tile objects
        tiles = []
        for p in paths:
            tiles.append(wsi_processing_pipeline.tile_extraction.tiles.Tile(tile_summary=None,
                                                   wsi_path=None,
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
                                                    t_p=None,
                                                    color_factor=None,
                                                    s_and_v_factor=None,
                                                    quantity_factor=None,
                                                    score=None,
                                                    tile_naming_func=None,
                                                    level=None,
                                                    best_level_for_downsample=None,
                                                    real_scale_factor=None,
                                                    roi=None,
                                                   patient_id=patient_id_getter(p), 
                                                    case_id=case_id_getter(p), 
                                                    slide_id=slide_id, 
                                                    classification_labels=classification_labels_getter(p), 
                                                    tile_path=p))
            
        return NamedObject(path=None, 
                           patient_id=patient_id_getter(p), 
                           case_id=case_id_getter(p), 
                           slide_id=slide_id_getter(p), 
                           classification_labels=classification_labels_getter(p), 
                           tiles=tiles)


def create_named_objects_from_tile_paths(tile_paths:List[pathlib.Path], 
                                            patient_id_getter:Callable, 
                                            case_id_getter:Callable, 
                                            slide_id_getter:Callable,                                                               
                                            classification_labels_getter:Callable,
                                            in_parallel:bool,                                            
                                            number_of_workers:int = multiprocessing.cpu_count())->List[NamedObject]:
        """
        Creates a list of NamedObject from a list of paths of prextracted tile images. Tiles from the same whole-slide
        image (with the same slide_ids) will be accumulated in one NamedObject.
        Arguments:
            tile_paths: List of paths to preextracted tiles
            patient_id_getter: Callable that takes a path and returns the corresponding patient_id
            case_id_getter: Callable that takes a path and returns the corresponding case_id
            slide_id_getter: Callable that takes a path and returns the corresponding slide_id
            classification_labels_getter: Callable that takes a path and returns the corresponding labels as a list
            in_parallel: if true, the work is performed on {number_of_workers} processes in parallel, else sequential
            number_of_workers: number of processes that shall be used (only relevant, if in_parallel is set to True)
        Returns:
            List of wsi_processing_pipeline.preprocessing.objects.NamedObject
        """
        
        #maps slide ids to list of tile paths
        d = {}
        for p in tile_paths:
            slide_id = slide_id_getter(p)
            if not(slide_id in d.keys()):
                d[slide_id] = []
            d[slide_id].append(p)
                
        named_objects = []
        
        if in_parallel:           
            pbar = tqdm(total=len(d))       
            def update(no):
                named_objects.append(no)
                pbar.update()
                
            def error(e):
                print(e)
                        
            with multiprocessing.Pool(number_of_workers) as pool:
                for slide_id in d.keys():
                    pool.apply_async(__create_named_object_by_slide_id, 
                                     kwds={"slide_id":slide_id,
                                           "tile_paths":d[slide_id],
                                           "patient_id_getter":patient_id_getter,
                                           "case_id_getter":case_id_getter, 
                                           "slide_id_getter":slide_id_getter, 
                                           "classification_labels_getter":classification_labels_getter}, 
                                           callback=update, 
                                           error_callback=error)
                    
                        
                pool.close()
                pool.join()
            
        else:    
            for slide_id in d.keys():
                named_objects.append(__create_named_object_by_slide_id(slide_id,
                                                d[slide_id],
                                                patient_id_getter, 
                                                case_id_getter, 
                                                slide_id_getter, 
                                                classification_labels_getter))
                
        return named_objects
    

    
##########################################
##########################################

class ObjectManager():
    def __init__(self, 
                 objects:List[NamedObject] = None):
        
        """

        """
        
        if not isinstance(objects, list):
            objects=[objects]
        self.objects=objects
        self.path=[items.path for items in self.objects]
        self.patient_id=[items.patient_id for items in self.objects]
        self.case_id=[items.case_id for items in self.objects]
        self.slide_id=[items.slide_id for items in self.objects]
        self.classification_labels=[items.classification_labels for items in self.objects]
        self.dataset_type=[items.dataset_type for items in self.objects]
        self.is_valid=[items.is_valid for items in self.objects]        
    
    def reset(self):
        self.path=[items.path for items in self.objects]
        self.patient_id=[items.patient_id for items in self.objects]
        self.case_id=[items.case_id for items in self.objects]
        self.slide_id=[items.slide_id for items in self.objects]
        self.classification_labels=[items.classification_labels for items in self.objects]
        self.dataset_type=[items.dataset_type for items in self.objects]
        self.is_valid=[items.is_valid for items in self.objects]    
 

    def get_all_top_tiles(self)->List[wsi_processing_pipeline.tile_extraction.tiles.Tile]:
        """
        Returns:
            returns all Tiles (that passed scoring in tile extraction process and are therefore relevant for training) 
            of all objects in self.objects combined in one list.
        """
        tile_list = []
        for o in self.objects:
            if not hasattr(o, 'tiles'):
                continue
            else:
                for t in o.tiles:
                    tile_list.append(t)
        return tile_list


    def convert_to_wsi_or_roi_object(self,
                                     save_tiles=False, 
                                     tile_score_thresh=0.55,
                                     return_as_tilesummary_object=True,
                                     tile_height=256,
                                     tile_width=256,
                                     tiles_folder_path=None,
                                     tile_naming_func=None):           
        """
            Convenience function that will deal with the process of tile extraction for you.
            Arguments:
                see wsi_processing_pipeline.tile_extraction.tiles.WsiOrROIToTilesMultithreaded()
        """
       
        
        wsi_path_to_wsi_info=create_WsiInfo(path=self.path,
                                               patient_id=self.patient_id,
                                               case_id=self.case_id, 
                                               slide_id=self.slide_id,
                                               classification_labels=self.classification_labels,
                                               dataset_type=self.dataset_type)
        
        tilesummaries=tiles.WsiOrROIToTilesMultithreaded(
                      wsi_paths=self.path, 
                      tiles_folder_path=tiles_folder_path, 
                      tile_height=tile_height,
                      tile_width=tile_width, 
                      tile_naming_func=tile_naming_func,
                      save_tiles=save_tiles, 
                      tile_score_thresh = tile_score_thresh, 
                      return_as_tilesummary_object=True,
                      wsi_path_to_wsi_info=wsi_path_to_wsi_info) 
        
        self.convert_to_wsi_or_roi_object_by_tilesummaries(tilesummaries=tilesummaries)
        
        
    def convert_to_wsi_or_roi_object_by_tilesummaries(self, 
                                                    tilesummaries:List[wsi_processing_pipeline.tile_extraction.tiles.TileSummary]):
        
        """
        If you prefer to create TileSummary objects yourself. (see wsi_processing_pipeline.tile_extraction example.ipynb)                                           
        """
        lst=[]
        for tilesummary in tilesummaries:
            objs = self.__find_named_objects__(tilesummary.wsi_path)
            if(len(objs) == 0):
                raise ValueError("No NamedObject in ObjectManager's object list, that has the same path as the tilesummary")
            if(len(objs) > 1):
                raise ValueError("More than one NamedObject in ObjectManager's object list, that has the same path as the \
                                 tilesummary. There should be only one NamedObject with the same path!")
            obj = objs[0]
            obj = WsiOrRoiObject(obj)            
            obj.tilesummary = tilesummary              
            tiles=[]
            for tile in obj.tilesummary.tiles:            
                tile.is_valid=obj.is_valid           
                tiles.append(tile)
            obj.tilesummary.tiles=tiles  
            obj.tiles=obj.tilesummary.top_tiles()
            lst.append(obj)    
            
        self.objects=lst 
        self.reset()
        
        #self.objects=sorted(self.objects, key=lambda x: x.path, reverse=True)
        #tilesummaries=sorted(tilesummaries, key=lambda x: x.wsi_path, reverse=True)
        #lst=[]
        #for summary, objects in zip(tilesummaries, self.objects):            
        #    objects=WsiOrRoiObject(objects)            
        #    objects.tilesummary = summary              
        #    ts=[]
        #    for t in objects.tilesummary.tiles:            
        #        t.is_valid=objects.is_valid           
        #        ts.append(t)
        #    objects.tilesummary.tiles=ts  
        #    objects.tiles=objects.tilesummary.top_tiles()
        #    lst.append(objects)    
        #    
        #self.objects=lst 
        #self.reset()
        
    def __find_named_objects__(self, path)->List[NamedObject]:
        """
        finds all NamedObject objects in self.objects by the specified path
        """
        objs = []
        for o in self.objects:
            if(o.path == path):
                objs.append(o)
        return objs
         
    def export_dataframe(self):
        if any(isinstance(el, WsiOrRoiObject) for el in self.objects):            
            fname=[]
            labels=[]
            is_valid=[]
            patient_id=[]
            case_id=[]
            slide_id=[]
            path=[]
            dataset_type=[]
                        
            for objects in self.objects: 
                for t in objects.tiles:
                    fname.append(t)
                    labels.append(t.classification_labels)
                    is_valid.append(t.is_valid)
                    patient_id.append(t.patient_id)
                    case_id.append(t.case_id)
                    slide_id.append(t.slide_id)
                    path.append(t.wsi_path)
                    dataset_type.append(t.dataset_type)
                
            df = pd.DataFrame(data={
            'fname':fname ,
            'labels':labels,
            'is_valid':is_valid,
            'patient_id':patient_id,
            'case_id':case_id,
            'slide_id':slide_id,    
            'dataset_type': dataset_type,             
            'path': path             
                })
        else:
            df = pd.DataFrame(data={
            'fname':self.path ,
            'labels':self.classification_labels,
            'is_valid': self.is_valid,
            'patient_id':self.patient_id,
            'case_id':self.case_id,
            'slide_id':self.slide_id,            
            'dataset_type':self.dataset_type,    
            
            })
            
        return df
    
    
    def split(self, splitter:Callable):
        """
        Arguments:
            splitter: a Callable, that takes the following input parameters (e.g. sklearn.model_selection.train_test_split):
                                    set of patient ids

                                    and returns two lists:
                                       list of patient ids for training
                                       list of patient ids for validation
        """
        if not any(el is None for el in self.patient_id):
            ids=self.patient_id
            itemid='patient_id'
        else:
            ids= self.case_id  
            itemid='case_id'
        
        # sorting ensures a reproduceable split of the ids. If the same ids are given in a different order to the method,
        # without sorting it would result in a different split, even if random_state is the the same (and numpy.random.seed()).
        ids.sort()
        
        ids_train, ids_test = splitter(list(set(ids)))              
        
        #print(len(ids_train)+len(ids_test))
        #print(len(ids_train))
        #print(len(ids_test))
        
        l=[] 
        
        for item in self.objects:
            if getattr(item, itemid) in ids_test:
                item.is_valid = True 
                item.dataset_type=DatasetType.validation
            else: 
                item.is_valid = False
                item.dataset_type=DatasetType.train
            if isinstance(item, WsiOrRoiObject):
                ts=[]
                for t in item.tilesummary.tiles:
                    t.patient_id=item.patient_id
                    t.case_id=item.case_id
                    t.slide_id=item.slide_id
                    t.dataset_type=item.dataset_type
                    t.is_valid=item.is_valid
                    t.classification_labels=item.classification_labels
                    ts.append(t)            
                item.tilesummary.tiles=ts  
                item.tiles=item.tilesummary.top_tiles()
            l.append(item)            
        self.objects=l
        self.reset()      
        

    def __split_KFold_cross_validation(self, 
                                       patient_ids:List[str],
                                       n_splits:int,
                                       current_iteration:int,
                                       random_state:int,
                                       shuffle:bool)->List[List[str]]:
        # sorting ensures a reproduceable split of the ids. If the same ids are given in a different order to the method,
        # without sorting it would result in a different split, even if random_state is the the same (and numpy.random.seed()).
        patient_ids.sort()        
        kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
        splits = list(kf.split(patient_ids))
        split_current_iteration = list(splits)[current_iteration]
        train_indices = split_current_iteration[0]
        val_indices = split_current_iteration[1]
        ids_train = [patient_ids[i] for i in train_indices]
        ids_val = [patient_ids[i] for i in val_indices]
        return ids_train, ids_val 

    def split_KFold_cross_validation(self, n_splits:int, current_iteration:int, random_state:int, shuffle:bool):   
        """
        Arguments:
            n_splits: number of splits == the k in k-fold
            current_iteration: index of the current split; e.g.: if you want to perform 5-fold crossvalidation, 
                                this parameter might be between [0,4]
            random_state: integer value, a random seed. If you keep this the same, the splitting will be 
                         consistent and always the same.
            shuffle: boolean value that indicates, if the ids should be shuffled before splitting

        """
        if current_iteration < 0 or current_iteration >= n_splits:
            raise ValueError(f'current_iteration must be in [0, {n_splits-1}]  (between 0 and n_splits-1)')
            
        splitter = functools.partial(self.__split_KFold_cross_validation, 
                                     n_splits=n_splits, 
                                     current_iteration=current_iteration, 
                                     random_state=random_state, 
                                     shuffle=shuffle)
        self.split(splitter)

        

##########################################
##########################################        
      
    
    
def create_WsiInfo(path:list,
                   patient_id:list,
                   case_id:list, 
                   slide_id:list,
                   classification_labels:list,
                   dataset_type:list):
    
    wsi_path_to_wsi_info={}
    for path,pat_id,case_id,slide_id,cls_lab,dat_typ in zip(path,
                                                        patient_id, 
                                                        case_id, 
                                                        slide_id,
                                                        classification_labels,
                                                        dataset_type):
                        
        wsi_info=tiles.WsiInfo(path = path, 
                               patient_id=pat_id,
                               case_id= case_id,
                               slide_id=slide_id,
                               classification_labels= cls_lab,
                               dataset_type=dat_typ, 
                               rois = None)        
        
            
        wsi_path_to_wsi_info[path]=wsi_info
        
    return  wsi_path_to_wsi_info   
        
class WsiOrRoiObject(NamedObject):                 
    def __init__(self,
                 no):
        self.no=no
        self.path=no.path 
        self.patient_id=no.patient_id 
        self.case_id=no.case_id
        self.slide_id=no.slide_id
        self.classification_labels=no.classification_labels
        self.dataset_type=no.dataset_type
        self.is_valid=no.is_valid
        self.tile_id=None
        
        super(WsiOrRoiObject, self).__init__(
                 path=self.path,
                 patient_id=self.patient_id, 
                 case_id=self.case_id, 
                 slide_id=self.slide_id,
                 classification_labels=self.classification_labels,
                 dataset_type=self.dataset_type)
        
    def process(self,
                save_tiles=False, 
                tile_score_thresh=0.55,
                return_as_tilesummary_object=True,
                tile_height=256,
                tile_width=256,
                tiles_folder_path=None,
                tile_naming_func=None):
        
        wsi_path_to_wsi_info=create_WsiInfo(path=[self.path],
                       patient_id=[self.patient_id],
                   case_id=[self.case_id], 
                   slide_id=[self.slide_id],
                   classification_labels=[self.classification_labels],
                   dataset_type=[self.dataset_type]
                      )        
        
        tilesummaries=tiles.WsiOrROIToTilesMultithreaded(
                      wsi_paths=[self.path], 
                      tiles_folder_path=tiles_folder_path, 
                      tile_height=tile_height,
                      tile_width=tile_width, 
                      tile_naming_func=tile_naming_func,
                      save_tiles=save_tiles, 
                      tile_score_thresh = tile_score_thresh, 
                      return_as_tilesummary_object=return_as_tilesummary_object,
                      wsi_path_to_wsi_info=wsi_path_to_wsi_info) 
        
               
        self.tilesummary=tilesummaries[0]
        
        ts=[]   
        
        for t in self.tilesummary.tiles:             
            t.is_valid=self.no.is_valid            
            ts.append(t)
        self.tilesummary.tiles=ts   
        
        self.tiles=[p for p in self.tilesummary.top_tiles()]