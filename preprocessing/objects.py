from wsi_processing_pipeline.tile_extraction.tiles import DatasetType
import sklearn
from enum import Enum

class NamedObject():    
    def __init__(self,                 
                 path=None,
                 patient_id=None, 
                 case_id=None, 
                 slide_id=None,
                 classification_label=None,
                 dataset_type:DatasetType=None):
        
        self.path=path
        self.patient_id=patient_id
        self.case_id=case_id
        self.slide_id=slide_id
        self.classification_label=classification_label
        self.dataset_type=dataset_type
        if isinstance(dataset_type, Enum):
            self.is_valid=True if dataset_type==DatasetType.validation else False 
        else:
            self.is_valid=None
        
    #def create(self, 
    #           path=None, 
    #           patient_id=None, 
    #           case_id=None, 
    #           slide_id=None,
    #           classification_label=None,
    #           dataset_type:DatasetType=None):
    #    self.path=path
    #    self.patient_id=patient_id
    #    self.case_id=case_id
    #    self.slide_id=slide_id
    #    self.classification_label=classification_label
    #    self.dataset_type=dataset_type
    #    if isinstance(dataset_type, Enum):
    #        self.is_valid=True if dataset_type==DatasetType.validation else False 
    #    else :
    #        self.is_valid=None
    #    return self
    
    def export_dataframe(self):
        df = pd.DataFrame(data={
        'path':self.path ,
        'patient_id':self.patient_id,
        'case_id':self.case_id,
        'slide_id':self.slide_id,
        'classification_label':self.classification_label,
        'dataset_type':self.dataset_type,
        'is_valid' : is_valid             
            }, index=[0])
        return df

class ObjectManager():
    def __init__(self, 
                 objects:list = None,
                 splitter:sklearn.model_selection.train_test_split=None,
                  ):
        if not isinstance(objects, list):
            objects=[objects]
        self.objects=objects
        self.splitter=splitter
        self.path=[items.path for items in self.objects]
        self.patient_id=[items.patient_id for items in self.objects]
        self.case_id=[items.case_id for items in self.objects]
        self.slide_id=[items.slide_id for items in self.objects]
        self.classification_label=[items.classification_label for items in self.objects]
        self.dataset_type=[items.dataset_type for items in self.objects]
        self.is_valid=[items.is_valid for items in self.objects]        
    
    def reset(self):
        self.path=[items.path for items in self.objects]
        self.patient_id=[items.patient_id for items in self.objects]
        self.case_id=[items.case_id for items in self.objects]
        self.slide_id=[items.slide_id for items in self.objects]
        self.classification_label=[items.classification_label for items in self.objects]
        self.dataset_type=[items.dataset_type for items in self.objects]
        self.is_valid=[items.is_valid for items in self.objects]    
    
    def convert_to_wsi_or_roi_object(self,
                                     save_tiles=False, 
                                     is_wsi=True, 
                                     tile_score_thresh=0.55,
                                     return_as_tilesummary_object=True,
                                     tile_height=256,
                                     tile_width=256,
                                     tiles_folder_path=None,
                                     tile_naming_func=None    ):           
        
       
        
        wsi_path_to_wsi_info=create_WsiInfo(path=self.path,
                       patient_id=self.patient_id,
                   case_id=self.case_id, 
                   slide_id=self.slide_id,
                   classification_labels=self.classification_label,
                   dataset_type=self.dataset_type
                      )
        tilesummaries=tiles.WsiOrROIToTilesMultithreaded(
                      wsi_paths=self.path, 
                      tiles_folder_path=tiles_folder_path, 
                      tile_height=tile_height,
                      tile_width=tile_width, 
                      tile_naming_func=tile_naming_func,
                      save_tiles=save_tiles, 
                      is_wsi=is_wsi,
                      tile_score_thresh = tile_score_thresh, 
                      return_as_tilesummary_object=return_as_tilesummary_object,
                      wsi_path_to_wsi_info=wsi_path_to_wsi_info)  
        self.objects=sorted(self.objects, key=lambda x: x.path, reverse=True)
        tilesummaries=sorted(tilesummaries, key=lambda x: x.wsi_path, reverse=True)
        lst=[]
        for summary, objects in zip(tilesummaries, self.objects):            
            objects=WsiOrRoiObject(objects)            
            objects.tilesummary = summary              
            ts=[]
            for t in objects.tilesummary.tiles:            
                t.is_valid=objects.is_valid           
                ts.append(t)
            objects.tilesummary.tiles=ts  
            objects.tiles=objects.tilesummary.top_tiles()
            lst.append(objects)    
            
        self.objects=lst 
        self.reset()
        
         
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
            'labels':self.classification_label,
            'is_valid': self.is_valid,
            'patient_id':self.patient_id,
            'case_id':self.case_id,
            'slide_id':self.slide_id,            
            'dataset_type':self.dataset_type,    
            
            })
            
        return df
    
    def split(self, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
        if not any(isinstance(el, NoneType) for el in self.patient_id):
            ids=self.patient_id
            itemid='patient_id'
        else:
            ids= self.case_id  
            itemid='case_id'
           
        ids_train, ids_test = self.splitter(list(set(ids)), 
                                       test_size=test_size,
                                       train_size=train_size,
                                       shuffle=shuffle, 
                                       stratify=stratify, 
                                       random_state=seed)              
        
        
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
                    t.classification_labels=item.classification_label
                    ts.append(t)            
                item.tilesummary.tiles=ts  
                item.tiles=item.tilesummary.top_tiles()
            l.append(item)            
        self.objects=l
        self.reset()      
        

def create_WsiInfo(path:list,
                   patient_id:list,
                   case_id:list, 
                   slide_id:list,
                   classification_labels:list,
                   dataset_type:list):
    
    wsi_path_to_wsi_info={}
    for pat,pat_id,cas_id,slid_id,cl_lab,dat_typ in zip(path,
                                                        patient_id, 
                                                        case_id, 
                                                        slide_id,
                                                        classification_labels,
                                                        dataset_type):
                        
        wsiinfo=tiles.WsiInfo(patient_id=pat_id,
                                                 case_id= cas_id,
                                                 slide_id=slid_id,
                                                 classification_labels= cl_lab,
                                                 dataset_type=dat_typ)        
        
            
        wsi_path_to_wsi_info[pat]=wsiinfo
        
    return  wsi_path_to_wsi_info   
        
class WsiOrRoiObject(NamedObject):
    
                  
    def __init__(self,
                 no,                          
                ):
        self.no=no
        self.path=no.path 
        self.patient_id=no.patient_id 
        self.case_id=no.case_id
        self.slide_id=no.slide_id
        self.classification_label=no.classification_label
        self.dataset_type=no.dataset_type
        self.is_valid=no.is_valid
        self.tile_id=None
        
        super(WsiOrRoiObject, self).__init__(
                 path=self.path,
                 patient_id=self.patient_id, 
                 case_id=self.case_id, 
                 slide_id=self.slide_id,
                 classification_label=self.classification_label,
                 dataset_type=self.dataset_type
                 )
        
    def process(self,
                save_tiles=False, 
                is_wsi=True, 
                tile_score_thresh=0.55,
                return_as_tilesummary_object=True,
                tile_height=256,
                tile_width=256,
                tiles_folder_path=None,
                tile_naming_func=None  ):
        
        wsi_path_to_wsi_info=create_WsiInfo(path=[self.path],
                       patient_id=[self.patient_id],
                   case_id=[self.case_id], 
                   slide_id=[self.slide_id],
                   classification_labels=[self.classification_label],
                   dataset_type=[self.dataset_type]
                      )        
        
        tilesummaries=tiles.WsiOrROIToTilesMultithreaded(
                      wsi_paths=[self.path], 
                      tiles_folder_path=tiles_folder_path, 
                      tile_height=tile_height,
                      tile_width=tile_width, 
                      tile_naming_func=tile_naming_func,
                      save_tiles=save_tiles, 
                      is_wsi=is_wsi,
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