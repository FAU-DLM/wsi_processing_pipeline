#https://stackoverflow.com/questions/46641078/how-to-avoid-circular-dependency-caused-by-type-hinting-of-pointer-attributes-in
from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from .patient import Patient


import fastai
import numpy as np


from typing import List, Callable, Tuple, Union
import pathlib
from pathlib import Path
Path.ls = lambda x: [p for p in list(x.iterdir()) if '.ipynb_checkpoints' not in p.name]

import shared
from shared.enums import DatasetType
import preprocessing

class Predictor:  
    learner:fastai.learner.Learner = None
    patient_manager: shared.patient_manager.PatientManager
    
    def __init__(self, 
                 learner:fastai.learner.Learner,
                 patient_manager:shared.patient_manager.PatientManager):
        """
        Arguments:
            This is a convenience class for mapping predictions on tile level to corresponding predictions on whole-slide
            image and case level.
               
            learner: fastai learner used for prediction
            patient_manager: 
        """
        if learner == None:
            raise ValueError("learner must not be None")
        if patient_manager == None:
            raise ValueError("patient_manager must not be None")
               
        self.learner = learner
        self.patient_manager = patient_manager
    
    def __predict(self, dataloader:fastai.data.core.TfmdDL):
        return self.learner.get_preds(ds_idx=-1, 
                                 dl=dataloader, 
                                 with_input=False, 
                                 with_decoded=False, 
                                 reorder=True, )
    
    def __build_dataloader(self, 
                           pred_type:shared.enums.PredictionType,
                           dataset_type:shared.enums.DatasetType,
                           tile_size:int, 
                           batch_size:int)->typing.Tuple[fastai.data.core.DataLoaders, List[shared.tile.Tile]]:
        block_x = None
        get_x = None
        if(pred_type == shared.enums.PredictionType.preextracted_tiles):
            block_x = fastai.vision.data.ImageBlock
            get_x = lambda x: x.tile_path
            
        elif(pred_type == shared.enums.PredictionType.tiles_on_the_fly):
            block_x = preprocessing.tile_image_block.TileImageBlock
            get_x=lambda x: x
            
        else:
            assert False
            
        data_pred = fastai.data.block.DataBlock(
        blocks=(block_x, fastai.data.block.MultiCategoryBlock),
        get_x=get_x, 
        get_y=lambda x: x.labels,
        # all tiles that shall be predicted will be in the train dataset => split func has to return false for them
        splitter=fastai.data.transforms.FuncSplitter(lambda x: not(x.get_dataset_type() == dataset_type)),
        item_tfms = fastai.vision.augment.Resize(tile_size),
        batch_tfms=[])
        tiles_to_predict = [t for t in self.patient_manager.get_all_tiles() if t.get_dataset_type() == dataset_type]        
        return data_pred.dataloaders(tiles_to_predict, bs=batch_size).train, tiles_to_predict
    
    def __set_preds(self, 
                    predictions:typing.Tuple[torch.Tensor, torch.Tensor], 
                    tiles_to_predict:List[shared.tile.Tile], 
                    vocab:fastai.data.transforms.CategoryMap):
        """
            sets the predictions for every shared.tile.Tile object
            
            Arguments:
                predictions: result of learner.get_preds(ds_idx=1, with_input=False, with_decoded=False)
                                Tuple with raw predictions at first place
                vocab: dataloader.vocab == list of classes in the order they appear in the predictions
        """
        assert len(predictions) == 2
        
        preds_raw = predictions[0]
        
        y_true_one_hot_encoded = None
        if(len(predictions) > 1):
            y_true_one_hot_encoded = predictions[1]
                
        assert len(preds_raw) == len(tiles_to_predict)
        
        for i in range(0, len(tiles_to_predict)):
            if(y_true_one_hot_encoded != None):
                tiles_to_predict[i].labels_one_hot_encoded = y_true_one_hot_encoded[i]
            preds_dict = {}
            for n, Class in enumerate(vocab):
                preds_dict[Class] = preds_raw[i][n]
            tiles_to_predict[i].predictions_raw = preds_dict
            
    def __buildDl_predict_set_preds(self, 
                                    pred_type:shared.enums.PredictionType.preextracted_tiles, 
                                    dataset_type:shared.enums.DatasetType, 
                                    tile_size:int, 
                                    batch_size:int):        
        dl_pred, tiles_to_predict = self.__build_dataloader(pred_type=pred_type,
                                                            dataset_type=dataset_type,
                                                             tile_size=tile_size, 
                                                             batch_size = batch_size)
        preds = self.__predict(dataloader = dl_pred)
        self.__set_preds(predictions=preds, tiles_to_predict=tiles_to_predict, vocab=dl_pred.vocab)
        
    
    
    def predict_on_tiles(self,
                         prediction_type: shared.enums.PredictionType,
                                      dataset_type:shared.enums.DatasetType, 
                                      tile_size:int, 
                                      batch_size:int):
        """
        Raw predictions will be saved in each Tile object in the attribute "predictions_raw".
        
        Arguments:
            prediction_type: - preextracted_tiles => use this, if the patient manager uses tiles, that have already been
                                extracted and saved to disc as images
                             - tiles_on_the_fly => use this, if the patient manager was created with tilesummary objects
                                                     that contain the regional information about each tile and the tiles
                                                    get extracted from the whole-slide images (or preextracted regions
                                                    of interest) on the fly during dataloading.
            dataset_type: only tiles with this dataset_type will be used for prediction
            tile_size: tiles will be resized to this size by the dataloader
            thresholds: threshold for each class between 0 and 1. If the predicted probability is higher than the
                        threshold, the tile will be labeled with that class.
        Results:
            
        """
               
        self.__buildDl_predict_set_preds(pred_type=prediction_type, 
                                        dataset_type=dataset_type, 
                                         tile_size=tile_size, 
                                         batch_size=batch_size)
        
    def get_classes(self):
        return self.learner.dls.vocab
    
    
    def __calculate_predictions_for_one_wsi_or_case(self, 
                                                    wsi_or_case:Union[shared.wsi.WholeSlideImage,shared.case.Case], 
                                                    thresholds_tile_level:Dict[str, float], 
                                                    thresholds_higher_level:Dict[str, float]):
        """
        After predict_on_tiles was called, this functions uses the predictions on tile level to calculate predictions for
        the given whole-slide image/case and saves it in the WholeSlideImage/Case object.
        Arguments:
            wsi_or_case: WholeSlideImage or Case object 
            thresholds: dictionary with the class names as key (class names can be obtained with self.get_classes()) and
                        thresholds (between 0 and 1) as values
                        
                        - thresholds_tile_level: predict_on_tiles() stored the raw prediction probabilities between 0 and 1
                                                    in each Tile object. These thresholds will be used to determine the
                                                    predicted classes for each Tile.
                        - thresholds_higher_level: example: The wsi/case has 100 tiles in total.
                                                            The threshold for class A is 0.5. If for 50 or 
                                                            more tiles class A is predicted, the wsi/case will also be 
                                                            labeled with that class.
        """        
        # checks
        for c in self.get_classes():
            if(c not in thresholds_tile_level.keys()):
                raise ValueError(f'{c} missing in the thresholds_tile_level dictionary\'s keys')
            if(c not in thresholds_higher_level.keys()):
                raise ValueError(f'{c} missing in the thresholds_higher_level dictionary\'s keys')
                
        assert thresholds_higher_level.keys() == thresholds_tile_level.keys()
        
        ##
        # iterate over all tiles and count how many tiles were predicted with a certain class
        ##
        tile_count = 0
        class_count = np.zeros(len(thresholds_tile_level))
        for tile in wsi_or_case.get_tiles():
            
            assert tile.predictions_raw != None
            assert tile.predictions_raw.keys() == thresholds_tile_level.keys()
            
            tile_count += 1
            class_count += tile.calculate_predictions_ohe(thresholds=thresholds_tile_level)
        
        ##
        # calculate the ration: tiles with that class / all tiles
        # and apply threshold
        ##
        preds_raw = class_count/tile_count
        preds_thresh = preds_raw >= np.array(list(thresholds_higher_level.values()))
        
        ##
        # convert it into more human friendly readable dictionary with classes as keys
        ##
        preds_raw_dict = {}
        preds_thresh_dict = {}
        for i in range(0, len(class_count)):
            class_name = list(thresholds_tile_level.keys())[i]
            preds_raw_dict[class_name] = preds_raw[i]
            preds_thresh_dict[class_name] = preds_thresh[i]
            
        wsi_or_case.predictions_raw = preds_raw_dict
        wsi_or_case.predictions_thresh = preds_thresh_dict
        
    
    def calculate_predictions_for_one_wsi(self, 
                                            wsi:shared.wsi.WholeSlideImage, 
                                            thresholds_tile_level:Dict[str, float], 
                                            thresholds_higher_level:Dict[str, float]):
        """
        After predict_on_tiles was called, this functions uses the predictions on tile level to calculate predictions for
        the given whole-slide image and saves it in the WholeSlideImage object.
        Arguments:
            wsi: WholeSlideImage
            thresholds: dictionary with the class names as key (class names can be obtained with self.get_classes()) and
                        thresholds (between 0 and 1) as values
                        
                        - thresholds_tile_level: predict_on_tiles() stored the raw prediction probabilities between 0 and 1
                                                    in each Tile object. These thresholds will be used to determine the
                                                    predicted classes for each Tile.
                        - thresholds_higher_level: example: The wsi has 100 tiles in total.
                                                            The threshold for class A is 0.5. If for 50 or 
                                                            more tiles class A is predicted, the wsi will also be 
                                                            labeled with that class.
        """
        self.__calculate_predictions_for_one_wsi_or_case(wsi_or_case=wsi, 
                                                         thresholds_tile_level=thresholds_tile_level, 
                                                        thresholds_higher_level=thresholds_higher_level)
    
    def calculate_predictions_for_one_case(self, 
                                            case:shared.case.Case, 
                                            thresholds_tile_level:Dict[str, float], 
                                            thresholds_higher_level:Dict[str, float]):
        """
        After predict_on_tiles was called, this functions uses the predictions on tile level to calculate predictions for
        the given case and saves it in the Case object.
        Arguments:
            case: Case
            thresholds: dictionary with the class names as key (class names can be obtained with self.get_classes()) and
                        thresholds (between 0 and 1) as values
                        
                        - thresholds_tile_level: predict_on_tiles() stored the raw prediction probabilities between 0 and 1
                                                    in each Tile object. These thresholds will be used to determine the
                                                    predicted classes for each Tile.
                        - thresholds_higher_level: example: The case has 100 tiles in total.
                                                            The threshold for class A is 0.5. If for 50 or 
                                                            more tiles class A is predicted, the case will also be 
                                                            labeled with that class.
        """
        self.__calculate_predictions_for_one_wsi_or_case(wsi_or_case=case, 
                                                         thresholds_tile_level=thresholds_tile_level, 
                                                        thresholds_higher_level=thresholds_higher_level)
        
    
    def calculate_predictions_up_to_case_level(self, 
                                                  dataset_type:shared.enums.DatasetType, 
                                                  thresholds_tile_level:Dict[str, float], 
                                                  thresholds_higher_level:Dict[str, float]):
        """
        After predict_on_tiles was called, this functions uses the predictions on tile level to calculate predictions up
        to the case level using the given thresholds.
        Arguments:
            dataset_type: only patients from that dataset will taken into account
            thresholds: dictionary with the class names as key (class names can be obtained with self.get_classes()) and
                        thresholds (between 0 and 1) as values
                        
                        - thresholds_tile_level: predict_on_tiles() stored the raw prediction probabilities between 0 and 1
                                                    in each Tile object. These thresholds will be used to determine the
                                                    predicted classes for each Tile.
                        - thresholds_higher_level: example: There is a case with 100 tiles in total.
                                                            The threshold for class A is 0.5. If for 50 or 
                                                            more tiles class A is predicted, the case will also be 
                                                            labeled with that class.
                                                    Same way of calculation on wsi level.
        """               
        for patient in self.patient_manager.patients:
            if(patient.dataset_type == dataset_type):
                for case in patient.cases:
                    self.calculate_predictions_for_one_case(case=case, 
                                                       thresholds_tile_level=thresholds_tile_level, 
                                                       thresholds_higher_level=thresholds_higher_level)
                    for wsi in case.whole_slide_images:
                         self.calculate_predictions_for_one_wsi(wsi=wsi, 
                                                       thresholds_tile_level=thresholds_tile_level, 
                                                       thresholds_higher_level=thresholds_higher_level)







        
        
        
        
        
#import fastai
#import wsi_processing_pipeline
#from wsi_processing_pipeline.preprocessing.objects import NamedObject
#import pandas as pd
#
#class Predictor(object):
#    
#    def __init__(self, 
#                 learner:fastai.learner.Learner,                  
#                 path, 
#                 tta:bool=False,
#                 thresh:float=0.5, 
#                 exclude_failed:bool=False, 
#                 dl:fastai.data.core.TfmdDL=None):
#        
#        if learner is None or not isinstance(learner, fastai.learner.Learner):            
#            raise AssertionError('please make sure to use a "fastai.learner.Learner" as "learner" input') 
#        
#        if dl is not None and not isinstance(dl, fastai.data.core.TfmdDL):        
#            raise AssertionError('please make sure to use a "fastai.data.core.TfmdDL" as dataloader - dl - input')
#        
#        elif dl is not None and isinstance(dl, fastai.data.core.TfmdDL):
#            self.dl = dl
#            self.ds = dl.dataset
#        
#        elif dl is None:
#            self.dl = learner.dls.valid
#            self.ds = learner.dls.valid_ds
#        else:
#            assert False # This case should not happen
#                    
#        self.learner=learner        
#        self.path=path
#        self.tta=tta
#        self.thresh=thresh
#        self.exclude_failed=exclude_failed
#                        
#        self.cat = fastai.data.transforms.Categorize(vocab=self.ds.vocab)
#        self.ds_items_checker()
#
#       
#    def ds_items_checker(self):        
#        if isinstance(self.ds.items, list):
#           
#            if any(isinstance(el, NamedObject) for el in self.ds.items) or any(isinstance(el, WsiOrRoiObject) for el in self.ds.items):
#                self.ds_items=self.ds.items#ObjectManager(self.ds.items).objects  
#                
#            elif any(isinstance(el, tiles.Tile) for el in self.ds.items):
#              
#                self.ds_items=self.ds.items
#               
#            else:
#                raise AssertionError('Items of a dataset should be of type list containing NamedObjects or TileObjects or of a pandas dataframe ')            
#                     
#        elif isinstance(self.ds.items, pd.core.frame.DataFrame ):
#                self.ds_items=self.ds.items
#                
#        else:
#            raise AssertionError('Items of a dataset should be of type list containing NamedObjects or TileObjects or of a pandas dataframe ')            
#                     
#        
#    def tile_id_checker(self):   
#        if self.tile_ids is None:
#            pass
#        
#        elif type(self.tile_ids) is list:            
#            pass
#
#        elif type(self.tile_ids) is str:
#            self.tile_ids=[self.tile_ids]
#            
#                      
#        elif isinstance(self.tile_ids, pathlib.PosixPath):     
#            self.tile_ids=[str(self.tile_ids)]
#                          
#        elif isinstance(self.tile_ids, NamedObject) or isinstance(self.tile_ids, WsiOrRoiObject):
#            self.tile_ids=[self.tile_ids]
#        elif isinstance(self.tile_ids, tiles.Tile):
#            self.tile_ids=[self.tile_ids]   
#        else:
#            raise AssertionError('tile_ids should be a string/PosixPath or a list of strings/PosixPaths containing the Path to the image')
#        
#        self.match_ds_items_and_tile_ids()
#    
#      
#    def match_ds_items_and_tile_ids(self):
#        if type(self.tile_ids) is list: 
#            if any(isinstance(el, NamedObject) for el in self.tile_ids) or any(isinstance(el, WsiOrRoiObject) for el in self.tile_ids):
#                
#                self.ds_items=[items for items in self.tile_ids if items in self.ds_items]
#            elif any(isinstance(el, tiles.Tile) for el in self.tile_ids):
#                self.ds_items=[items for items in self.tile_ids if items in self.ds_items]
#                
#            elif any(isinstance(el, str) for el in self.tile_ids):                               
#                self.ds_items=self.ds_items[self.ds_items.fname.isin(self.tile_ids)]
#                
#            elif any(isinstance(el, pathlib.PosixPath) for el in self.tile_ids):                
#                self.ds_items=self.ds_items[self.ds_items.fname.isin(self.tile_ids)]
#            
#            
#                
#        if self.tile_ids is None:
#            pass  
#   
#
#    def get_prediction_per_tile(self, ids=None):
#        
#        self.tile_ids=ids        
#        self.tile_id_checker()        
#        
#        cls=[]
#        prob=[]        
#        above_thr=[]        
#        
#        if isinstance(self.ds_items, pd.DataFrame):
#           
#            tile_ids=[]
#            slide_ids=[]
#            case_ids=[]
#            labels=[]
#            for ids, item in tqdm(self.ds_items.iterrows(),  total=self.ds_items.shape[0]):        
#                    tile_id=item['fname']
#                    tile_ids.append(tile_id)            
#                    slide_ids.append(item['slide_id'])
#                    case_ids.append(item['case_id'])
#                    labels.append(item['labels'])
#            true_cls=labels
#            
#        else:
#            try:
#                tile_ids=[obj.path for obj in self.ds_items]
#            except:
#                tile_ids=[obj.wsi_path for obj in self.ds_items]
#            true_cls=[obj.classification_labels for obj in self.ds_items]
#            case_ids=[obj.case_id for obj in self.ds_items]
#            slide_ids=[obj.slide_id for obj in self.ds_items]
#           
#        if self.tile_ids is not None: 
#            
#            if isinstance(self.ds_items, pd.DataFrame):
#            
#                df=pd.DataFrame(data={'fname':tile_ids, 'labels':labels})           
#                if len(df)>=9:
#                    for i in range (1,12):                    
#                        if len(df)%i == 0:                        
#                            bs=i
#                else: bs=len(df)        
#                dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
#                                   get_x=ColReader('fname') , 
#                                   get_y=ColReader('labels') ,   
#                                   splitter=RandomSplitter(valid_pct=0),
#                                  
#                                   batch_tfms=[*aug_transforms(mult=1.0, 
#                                                             do_flip=True, 
#                                                             flip_vert=True, 
#                                                             max_rotate=10.0, 
#                                                             max_zoom=1.1, 
#                                                             max_lighting=0.2, 
#                                                             max_warp=0.2, 
#                                                             p_affine=0.75, 
#                                                             p_lighting=0.75, 
#                                                             xtra_tfms=None, 
#                                                             size=None, 
#                                                             mode='bilinear', 
#                                                             pad_mode='reflection', 
#                                                             align_corners=True, 
#                                                             batch=True, 
#                                                             min_scale=1.0), 
#                                             Normalize.from_stats(*imagenet_stats)
#                                            
#                                            ]
#                                           )  
#                self.dl=dblock.dataloaders(df, bs=bs)[0]
#                
#            else:
#                if len(self.ds_items)>=9:
#                    for i in range (1,12):                    
#                        if len(self.ds_items)%i == 0:                        
#                            bs=i
#                else: bs=len(self.ds_items)
#                t_dblock = DataBlock(
#                    blocks=(TileImageBlock, CategoryBlock),                   
#                    get_x=lambda x: x, 
#                    get_y=lambda x: x.classification_labels,
#                    splitter=RandomSplitter(valid_pct=0),
#                    item_tfms=Resize(224),
#                    batch_tfms=[*aug_transforms(mult=1.0, 
#                                                             do_flip=True, 
#                                                             flip_vert=True, 
#                                                             max_rotate=10.0, 
#                                                             max_zoom=1.1, 
#                                                             max_lighting=0.2, 
#                                                             max_warp=0.2, 
#                                                             p_affine=0.75, 
#                                                             p_lighting=0.75, 
#                                                             xtra_tfms=None, 
#                                                             size=None, 
#                                                             mode='bilinear', 
#                                                             pad_mode='reflection', 
#                                                             align_corners=True, 
#                                                             batch=True, 
#                                                             min_scale=1.0), 
#                                             Normalize.from_stats(*imagenet_stats)                                            
#                                            ]
#                    )
#            
#                self.dl=t_dblock.dataloaders(self.ds_items, bs=bs)[0]
#                try:
#                    tile_ids=[obj.path for obj in self.ds_items]
#                except:
#                    tile_ids=[obj.wsi_path for obj in self.ds_items]
#                true_cls=[obj.classification_labels for obj in self.ds_items]
#                case_ids=[obj.case_id for obj in self.ds_items]
#                slide_ids=[obj.slide_id for obj in self.ds_items]
#        
#        
#        
#        if self.tta is False:
#            prbs,_=self.learner.get_preds(dl=self.dl)
#
#        else:            
#            
#            prbs, _=self.learner.tta(dl=self.dl, use_max=False)  
#        
#        
#        for prb in prbs:
#            prb=prb.numpy()
#            
#            cl=np.argmax(prb)
#            prob.append(prb)
#            
#            cl = self.cat.decodes(cl)
#            cls.append(cl)
#            above_thresh=False
#
#            if prb[np.argmax(prb)] > self.thresh:
#                above_thresh=True
#            above_thr.append(above_thresh)    
#        
#        tile_dataframe=pd.DataFrame(data={'tile_id':tile_ids,
#                                     'slide_id':slide_ids,
#                                     'case_id':case_ids,
#                                     'predicted_class':cls,
#                                     'true_class':true_cls,    
#                                     'probabilities':prob,
#                                     'above_threshold': above_thr,
#                                     }
#                               )
#        
#        if not isinstance(self.ds_items, pd.DataFrame):
#            for obj,cl,pr,ab in zip(self.ds_items,cls,prob,above_thr):
#                obj.predicted_class=cl
#                obj.probabilities=pr
#                obj.above_threshold=ab
#            
#        return tile_dataframe 
#
#    
#    def get_prediction_per_slide(self, ids=None):
#        
#        slide_ids=ids
#        
#        if slide_ids:                
#            if type(slide_ids) is not list:
#                try: 
#                    slide_ids=[slide_ids]
#                except:     
#                    raise NotImplementedError
#            
#            ids=[] 
#            
#            if isinstance(self.ds_items, pd.DataFrame):
#                for iid,item in zip(self.ds_items['slide_id'].to_list(),self.ds_items['fname'].to_list()):
#                    
#                    if iid in slide_ids:
#                        ids.append(item)
#                        
#            if isinstance(self.ds_items, list):
#                for objects in self.ds_items:
#                    
#                    if objects.slide_id in slide_ids:
#                        ids.append(objects)
#                    
#                        
#                
#            if ids==[]:
#                raise ValueError('None of your list of slide ids is contained within the provided dataset; please check your data!')
#                
#
#        
#        tile_dataframe=self.get_prediction_per_tile(ids=ids)
#        
#        
#        df_list=[]
#        for unique_item in tile_dataframe['slide_id'].unique():
#            
#            classlabel=[]
#            ass_prob=[]
#            ids=[]
#            for dat, thr, iid,true_cls, case_id in zip(tile_dataframe.loc[tile_dataframe['slide_id'] == unique_item, 'probabilities'],
#                                tile_dataframe.loc[tile_dataframe['slide_id'] == unique_item, 'above_threshold'],
#                                tile_dataframe.loc[tile_dataframe['slide_id'] == unique_item, 'tile_id'],
#                                tile_dataframe.loc[tile_dataframe['slide_id'] == unique_item, 'true_class'],
#                                tile_dataframe.loc[tile_dataframe['slide_id'] == unique_item, 'case_id']):        
#                
#                if self.exclude_failed:
#                    if thr > self.thresh:
#
#                        classlabel.append(self.cat.decodes(np.argmax(dat)))
#                        ass_prob.append(dat[np.argmax(dat)])
#                else:
#                    classlabel.append(self.cat.decodes(np.argmax(dat)))
#                    ass_prob.append(dat[np.argmax(dat)])
#
#                ids.append(iid)
#
#            cls_slide_lbl_counts=pd.DataFrame({'classlabel':classlabel})['classlabel'].value_counts().to_list()
#            cls_slide_lbl=pd.DataFrame({'classlabel':classlabel})['classlabel'].value_counts().idxmax()
#
#            idxmin=pd.DataFrame({'ass_prob':ass_prob})['ass_prob'].idxmin()
#            idxmax=pd.DataFrame({'ass_prob':ass_prob})['ass_prob'].idxmax()
#            min_slide_prob=pd.DataFrame({'ass_prob':ass_prob})['ass_prob'][idxmin]
#            max_slide_prob=pd.DataFrame({'ass_prob':ass_prob})['ass_prob'][idxmax]
#
#
#            overall_slide_label_probs=(pd.DataFrame({'classlabel':classlabel})['classlabel'].value_counts()/pd.DataFrame({'classlabel':classlabel})['classlabel'].count()).to_list()
#
#            ass_overall_slide_label_prob=(pd.DataFrame({'classlabel':classlabel})['classlabel'].value_counts()/pd.DataFrame({'classlabel':classlabel})['classlabel'].count())[cls_slide_lbl]
#
#            above_thresh=False
#            if ass_overall_slide_label_prob > self.thresh:
#                above_thresh=True
#
#           
#            slide_dataframe=pd.DataFrame(data={
#                               'case_id': case_id,
#                               'slide_id': unique_item,
#                               'predicted_class_slide_label':cls_slide_lbl,
#                               'true_class':true_cls, 
#                               'associated_overall_slide_label_probability': ass_overall_slide_label_prob,
#                               'above_threshold':above_thresh,
#                               'class_slide_label_counts':[cls_slide_lbl_counts],                  
#                               'overall_slide_label_probabilities' :[overall_slide_label_probs],
#                               'minimum_slide_label_prob':min_slide_prob,
#                               'maximum_slide_label_prob':max_slide_prob,                           
#
#                              })   
#            
#            df_list.append(slide_dataframe)
#
#        slide_dataframe = pd.concat(df_list)
#
#
#
#        return tile_dataframe, slide_dataframe
#
#
#    def get_prediction_per_case(self, ids=None , tile_level=True):
#
#        case_ids=ids
#        slide_ids=None
#        if case_ids:                
#            if type(case_ids) is not list:
#                try: 
#                    case_ids=[case_ids]
#                except:     
#                    raise NotImplementedError
#
#
#            slide_ids=[]
#            if isinstance(self.ds_items, pd.DataFrame):
#                for iid,item in zip(self.ds_items['case_id'].to_list(),self.ds_items['fname'].to_list()):
#                    
#                    if iid in slide_ids:
#                        ids.append(item)
#                        
#            if isinstance(self.ds_items, list):
#                for objects in self.ds_items:
#                    
#                    if objects.case_id in case_ids:
#                        ids.append(objects)     
#            
#           
#
#            if case_ids==[] and slide_ids==[]:
#                raise ValueError('None of your list of case ids is contained within the provided dataset; please check your data!')
#
#
#
#
#        tile_dataframe,slide_dataframe=self.get_prediction_per_slide(
#                                                       ids=slide_ids 
#                                                       )      
#
#
#
#        if tile_level:
#            dataframe=tile_dataframe
#            probability= 'probabilities'
#            pred_label='predicted_class'
#        else:
#            dataframe=slide_dataframe
#            probability= 'associated_overall_slide_label_probability'
#            pred_label= 'predicted_class_slide_label'
#        df_list=[]
#        for unique_item in slide_dataframe['case_id'].unique():
#            classlabel=[]
#            ass_prob=[]
#            ids=[]
#
#            for label, prob, thr, iid  in zip(dataframe.loc[dataframe['case_id'] == unique_item, pred_label],
#                        dataframe.loc[dataframe['case_id'] == unique_item, probability],
#                         dataframe.loc[dataframe['case_id'] == unique_item, 'above_threshold'] , 
#                         dataframe.loc[dataframe['case_id'] == unique_item, 'true_class']           ):  
#                if tile_level:
#                    prob=prob[np.argmax(prob)]
#
#                if self.exclude_failed:
#                    if thr > self.thresh:
#                        classlabel.append(label)
#                        ass_prob.append(prob)
#                else:
#                    classlabel.append(label)
#                    ass_prob.append(prob)
#
#                ids.append(iid)    
#            cls_case_lbl_counts=pd.DataFrame({'classlabel':classlabel})['classlabel'].value_counts().to_list()
#            cls_case_lbl=pd.DataFrame({'classlabel':classlabel})['classlabel'].value_counts().idxmax()    
#
#            idxmin=pd.DataFrame({'ass_prob':ass_prob})['ass_prob'].idxmin()
#            idxmax=pd.DataFrame({'ass_prob':ass_prob})['ass_prob'].idxmax()
#            min_case_prob=pd.DataFrame({'ass_prob':ass_prob})['ass_prob'][idxmin]
#            max_case_prob=pd.DataFrame({'ass_prob':ass_prob})['ass_prob'][idxmax]
#
#            overall_case_label_probs=(pd.DataFrame({'classlabel':classlabel})['classlabel'].value_counts()/pd.DataFrame({'classlabel':classlabel})['classlabel'].count()).to_list()
#
#            ass_overall_case_label_prob=(pd.DataFrame({'classlabel':classlabel})['classlabel'].value_counts()/
#                                         pd.DataFrame({'classlabel':classlabel})['classlabel'].count())[cls_case_lbl]
#
#
#            above_thresh=False
#            if ass_overall_case_label_prob > self.thresh:
#                above_thresh=True
#            true_cls=ids[0]
#            case_dataframe=pd.DataFrame(data={
#                                   'case_id': unique_item,
#                                   'predicted_class_case_label':cls_case_lbl,
#                                   'true_class':true_cls,
#                                   'associated_overall_case_label_probability': ass_overall_case_label_prob,
#                                   'above_threshold':above_thresh,
#                                   'class_case_label_counts':[cls_case_lbl_counts],                  
#                                   'overall_case_label_probabilities' :[overall_case_label_probs],
#                                   'minimum_case_label_prob':min_case_prob,
#                                   'maximum_case_label_prob':max_case_prob,        
#
#                                  })   
#
#            df_list.append(case_dataframe)
#
#        case_dataframe = pd.concat(df_list)      
#    
#        return tile_dataframe, slide_dataframe, case_dataframe