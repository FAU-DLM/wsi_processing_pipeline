#https://stackoverflow.com/questions/46641078/how-to-avoid-circular-dependency-caused-by-type-hinting-of-pointer-attributes-in
from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from .patient import Patient


import torch
import fastai
import numpy
import numpy as np


from typing import List, Callable, Tuple, Union
import pathlib
from pathlib import Path
Path.ls = lambda x: [p for p in list(x.iterdir()) if '.ipynb_checkpoints' not in p.name]

import shared
from shared.enums import DatasetType
import preprocessing
import pandas as pd
from tqdm import tqdm
import multiprocessing

class Predictor:  
    learner:fastai.learner.Learner = None
    patient_manager: shared.patient_manager.PatientManager = None
    
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
    
    def __one_hot_encode(self, vocab:List[Union[str, int]], labels:List[Union[str, int]])->numpy.ndarray:
        """
        vocab: List of all possible classes (strings or ints)
        labels: labels of the object
        """
        for l in labels:
            if(l not in vocab):
                raise ValueError(f'{l} is in labels but not in vocab.')
                
        ohe_labels = np.zeros(len(vocab), dtype=np.int0)
        for n, v in enumerate(vocab):
            if(v in labels):
                ohe_labels[n] = 1
        return ohe_labels
            
    def __build_dataloader(self, 
                           pred_type:shared.enums.PredictionType,
                           dataset_type:shared.enums.DatasetType,
                           tile_size:int, 
                           batch_size:int)->typing.Tuple[fastai.data.core.DataLoaders, List[shared.tile.Tile]]:
        ###
        # Option 1: build complete new Dataloaders
        ###
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
        tiles_to_predict = self.patient_manager.get_tiles(dataset_type = dataset_type)
        dataloader = data_pred.dataloaders(tiles_to_predict, bs=batch_size).train
        #https://forums.fast.ai/t/get-preds-returns-smaller-than-expected-tensor/47519
        dataloader.drop_last = False
        dataloader.shuffle = False
        
        
        ###
        # Option 2: build test Dataloader from existing learner.dls
        ###
        #tiles_to_predict = self.patient_manager.get_tiles(dataset_type = dataset_type)
        #dataloader = self.learner.dls.test_dl(tiles_to_predict)
        
        return dataloader, tiles_to_predict
    
    
    def __set_preds(self, 
                    predictions:typing.Tuple[torch.Tensor, torch.Tensor], 
                    tiles_to_predict:List[shared.tile.Tile], 
                    vocab:fastai.data.transforms.CategoryMap):
        """
            sets the raw predictions (== predicted probabilities for each class) for every shared.tile.Tile object
            
            Arguments:
                predictions: result of learner.get_preds(ds_idx=1, with_input=False, with_decoded=False)
                                Tuple with raw predictions at first place
                vocab: dataloader.vocab == list of classes in the order they appear in the predictions
        """
        assert len(predictions) == 3
        
        preds_raw = predictions[0]
        
        y_true_one_hot_encoded = None
        if(len(predictions) > 1):
            y_true_one_hot_encoded = predictions[1]
                
        assert len(preds_raw) == len(tiles_to_predict)
        
        for i in range(0, len(tiles_to_predict)):
            if(y_true_one_hot_encoded is not None):
                tiles_to_predict[i].labels_one_hot_encoded = numpy.array(y_true_one_hot_encoded[i]).astype(np.int0)
            preds_dict = {}
            for n, Class in enumerate(vocab):
                preds_dict[Class] = preds_raw[i][n].item()
            tiles_to_predict[i].predictions_raw = preds_dict
            tiles_to_predict[i].loss = predictions[2][i].item()
            
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
        
    #def __predict(self, dataloader:fastai.data.core.TfmdDL):
    #    return self.learner.get_preds(ds_idx=-1, 
    #                             dl=dataloader, 
    #                             with_input=False, 
    #                             with_decoded=False, 
    #                             reorder=True, 
    #                             with_loss=True)
    
    def __predict(self, dataset_type:shared.enums.DatasetType, tile):
        tile.labels_one_hot_encoded = self.__one_hot_encode(vocab=self.get_classes(), labels=tile.get_labels())
        
        _,_, preds_raw = self.learner.predict(tile)
        predictions_raw = {}
        for Class, pred_raw in zip(self.get_classes(), preds_raw):
            predictions_raw[Class] = pred_raw            
        tile.predictions_raw = predictions_raw
   
        tile.loss = self.learner.loss_func(preds_raw, torch.tensor(tile.labels_one_hot_encoded))  
    
    def predict_on_tiles(self,
                         prediction_type: shared.enums.PredictionType,
                         dataset_type:shared.enums.DatasetType, 
                         tile_size:int, 
                         batch_size:int):
        """
        Raw predictions (== predicted probabilities for each class) will be saved in each Tile object 
        in the attribute "predictions_raw". No thresholds are applied here.
        
        Arguments:
            prediction_type: - preextracted_tiles => use this, if the patient manager uses tiles, that have already been
                                extracted and saved to disc as images
                             - tiles_on_the_fly => use this, if the patient manager was created with tilesummary objects
                                                     that contain the regional information about each tile and the tiles
                                                    get extracted from the whole-slide images (or preextracted regions
                                                    of interest) on the fly during dataloading.
            dataset_type: only tiles with this dataset_type will be used for prediction
            tile_size: tiles will be resized to this size by the dataloader
        Results:
            
        """
        ###
        # There is currently a bug in learner.get_preds() which leads to completely wrong predictions
        # learner.predict is a lot slower, but it gives correct predictions
        ###
        #self.__buildDl_predict_set_preds(pred_type=prediction_type, 
        #                                dataset_type=dataset_type, 
        #                                 tile_size=tile_size, 
        #                                 batch_size=batch_size)
        
        tiles_to_predict = self.patient_manager.get_tiles(dataset_type = dataset_type)
        for t in tqdm(tiles_to_predict):
            self.__predict(dataset_type=dataset_type, tile=t)
        #pbar = tqdm(total=len(tiles_to_predict))
        #def update(res):
        #    pbar.update()
        #    
        #def error(e):
        #    print(e)
        #
        #with multiprocessing.Pool(processes=None) as pool:
        #    for t in tiles_to_predict:
        #        pool.apply_async(self.__predict, 
        #                         kwds={"dataset_type":dataset_type,
        #                               "tile":t}, 
        #                               callback=update, 
        #                               error_callback=error)
        #        
        #            
        #    pool.close()
        #    pool.join()

        
    def get_classes(self):
        return self.learner.dls.vocab
    
    
    
    def calculate_predictions_for_one_tile(self, 
                                           tile:shared.tile.Tile, 
                                           thresholds:Dict[str, float]):
        """
        After predict_on_tiles was called, this functions applies the given thresholds to calculate the predicted classes
        and stores it into the tile's predictions_thresh attribute.
        """
        # checks
        for c in self.get_classes():
            if(c not in thresholds.keys()):
                raise ValueError(f'{c} missing in the thresholds_tile_level dictionary\'s keys')
                
        predictions_thresh = {}
        for Class in thresholds.keys():
            predictions_thresh[Class] = tile.predictions_raw[Class] >= thresholds[Class]
        tile.predictions_thresh = predictions_thresh
    
    
    def __calculate_predictions_for_one_wsi_or_case(self, 
                                                    wsi_or_case:Union[shared.wsi.WholeSlideImage,shared.case.Case],  
                                                    thresholds:Dict[str, float]):
        """
        After predict_on_tiles was called, this functions uses the predictions on tile level to calculate predictions for
        the given whole-slide image/case and saves it in the WholeSlideImage/Case object.
        The raw predictions of the wsi's/case's tiles are summed up for each class and divided by the number of tiles. This results
        in the raw predictions for the wsi/case.
        Then the <thresholds> are applied to get the predicted classes.
        Arguments:
            wsi_or_case: WholeSlideImage or Case object 
            thresholds: dictionary with the class names as key (class names can be obtained with self.get_classes()) and
                        thresholds (between 0 and 1) as values
                        
        """        
        # checks
        for c in self.get_classes():
            if(c not in thresholds.keys()):
                raise ValueError(f'{c} missing in the thresholds dictionary\'s keys')
                        
        ##
        # iterate over all tiles and sums up all raw predictions for each class
        ##
        tile_count = 0
        # old version: thresholded preds (yes or no) over all tiles were summed up
        #class_count = np.zeros(len(thresholds_tile_level))
        
        # new version: raw predictions over all tiles are summed up
        summed_up_raw_preds = np.zeros(len(thresholds))
        for tile in wsi_or_case.get_tiles():
            
            assert tile.predictions_raw != None
            assert tile.predictions_raw.keys() == thresholds.keys()
            
            tile_count += 1
            
            # old version: thresholded preds (yes or no) over all tiles were summed up
            #class_count += tile.calculate_predictions_ohe(thresholds=thresholds_tile_level)
            
            # new version: raw predictions over all tiles are summed up
            summed_up_raw_preds += np.array(list(tile.predictions_raw.values()))
        
        ##
        # divide summed up raw predictions by number of tiles
        # and apply threshold
        ##
        preds_raw = summed_up_raw_preds/tile_count
        preds_thresh = preds_raw >= np.array(list(thresholds.values()))
        
        ##
        # convert it into more human friendly readable dictionary with classes as keys
        ##
        preds_raw_dict = {}
        preds_thresh_dict = {}
        for i in range(0, len(thresholds)):
            class_name = list(thresholds.keys())[i]
            preds_raw_dict[class_name] = preds_raw[i]
            preds_thresh_dict[class_name] = preds_thresh[i]
            
        wsi_or_case.predictions_raw = preds_raw_dict
        wsi_or_case.predictions_thresh = preds_thresh_dict
        
    
    def calculate_predictions_for_one_wsi(self, 
                                            wsi:shared.wsi.WholeSlideImage, 
                                            thresholds:Dict[str, float]):
        """
        After predict_on_tiles was called, this functions uses the predictions on tile level to calculate predictions for
        the given whole-slide image and saves it in the WholeSlideImage object.
        The raw predictions of the wsi's tiles are summed up for each class and divided by the number of tiles. This results
        in the raw predictions for the wsi.
        Then the <thresholds> are applied to get the predicted classes.
        Arguments:
            wsi: WholeSlideImage 
            thresholds: dictionary with the class names as key (class names can be obtained with self.get_classes()) and
                        thresholds (between 0 and 1) as values
        """
        self.__calculate_predictions_for_one_wsi_or_case(wsi_or_case=wsi,  
                                                         thresholds=thresholds)
    
    def calculate_predictions_for_one_case(self, 
                                            case:shared.case.Case, 
                                            thresholds:Dict[str, float]):
        """
        After predict_on_tiles was called, this functions uses the predictions on tile level to calculate predictions for
        the given case and saves it in the Case object.
        The raw predictions of the case's tiles are summed up for each class and divided by the number of tiles. This results
        in the raw predictions for the case.
        Then the <thresholds> are applied to get the predicted classes.
        Arguments:
            case: Case object 
            thresholds: dictionary with the class names as key (class names can be obtained with self.get_classes()) and
                        thresholds (between 0 and 1) as values
        """
        self.__calculate_predictions_for_one_wsi_or_case(wsi_or_case=case,  
                                                        thresholds=thresholds)
        
    
    def calculate_predictions_up_to_case_level(self, 
                                                  dataset_type:shared.enums.DatasetType, 
                                                  thresholds:Dict[str, float]):
        """
        After predict_on_tiles was called, this functions uses the predictions on tile level to calculate predictions for
        the given whole-slide image/case and saves it in the WholeSlideImage/Case object.
        The raw predictions of the wsi's/case's tiles are summed up for each class and divided by the number of tiles. This results
        in the raw predictions for the wsi/case.
        Then the <thresholds> are applied to get the predicted classes.
        Arguments:
            wsi_or_case: WholeSlideImage or Case object 
            thresholds: dictionary with the class names as key (class names can be obtained with self.get_classes()) and
                        thresholds (between 0 and 1) as values
        """               
        for patient in self.patient_manager.get_patients(dataset_type=dataset_type):
            for case in patient.get_cases():
                self.calculate_predictions_for_one_case(case=case, 
                                                       thresholds=thresholds)
                for wsi in case.get_whole_slide_images():
                    self.calculate_predictions_for_one_wsi(wsi=wsi, 
                                                       thresholds=thresholds)
                    for roi in wsi.get_regions_of_interest():
                        for tile in roi.get_tiles():
                            self.calculate_predictions_for_one_tile(tile=tile, thresholds=thresholds)
    
    def export_dataframe(self, 
                         dataset_type:shared.enums.DatasetType, 
                         level:shared.enums.DataframeLevel)->pd.DataFrame:
        """
        Creates a dataframe for all patients from the specified dataset type with their saved predictions.
        Arguments:
            dataset_type:
            level:
        Returns:
            pandas.Dataframe
        """
       
        df = pd.DataFrame()
               
        if(level == shared.enums.DataframeLevel.case_level):
            for patient in self.patient_manager.get_patients(dataset_type=dataset_type):
                for case in patient.cases:
                    preds = []
                    for Class, bool_value in case.predictions_thresh.items():
                        if(bool_value):
                            preds.append(Class)
                            
                    df = df.append({'patient id' : patient.patient_id, 
                               'case id' : case.case_id,
                               'labels' : case.get_labels(),
                               'predictions' : preds, 
                               'predictions (tiles with class/all tiles)' : case.predictions_raw}, 
                              ignore_index=True)
                    
        elif(level == shared.enums.DataframeLevel.slide_level):
            for patient in self.patient_manager.get_patients(dataset_type=dataset_type):
                for case in patient.cases:
                    for wsi in case.whole_slide_images:
                        preds = []
                        for Class, bool_value in wsi.predictions_thresh.items():
                            if(bool_value):
                                preds.append(Class)
                            
                        df = df.append({'patient id' : patient.patient_id, 
                                   'case id' : case.case_id,
                                   'slide id' : wsi.slide_id,
                                   'labels' : wsi.get_labels(),
                                   'predictions' : preds, 
                                   'predictions (tiles with class/all tiles)' : wsi.predictions_raw}, 
                                  ignore_index=True)
        
        else:
            assert False
        
        return df
    
    
    def predict_on_tiles_KFold_cross_validation(self, 
                                       k:int, 
                                       seed:int, 
                                       iteration_to_weights:Dict[int, pathlib.Path], 
                                       prediction_type:shared.enums.PredictionType, 
                                       tile_size:int, 
                                       batch_size:int):
        """
        When you have done k-fold cross validation, you have k models and k non overlapping validation sets.
        This method iterates k times and for each iteration splits the dataset into train and validation
        loads the associated model weights and makes predictions
        on the associated validation set the specific model was not trained on. So in the end you have predictions
        for the complete dataset but always with a model that had not been trained on the data used for prediction.
        Raw predictions (== predicted probabilities for each class) will be saved in each Tile object 
        in the attribute "predictions_raw". No thresholds are applied here.
        
        
        Arguments:
            k: the k in k-fold
            seed: random seed, it is very important that you use the same seed you used for the split during training, so
                    that this method can recreate the exact same k splits. Otherwise predictions will very likely be made on
                    tiles that the model had already seen during training and would therefore be pointless.
            iteration_to_weights: A dictionary with the numbers of iteration as keys and the associated model weights
                                    as values.
            prediction_type:
            tile_size:
            batch_size:
        Result:
            Raw predictions are stored inside the Tile objects in the patient_manager.
        """
        if(k != len(iteration_to_weights)):
            raise ValueError('There have to be k models in iteration_to_weights')
        
        for current_iteration in tqdm(range(0, k)):
            ##
            # split dataset according to current iteration
            ##
            self.patient_manager.split_KFold_cross_validation(n_splits=k, 
                                             current_iteration=current_iteration, 
                                             random_state = seed, 
                                             shuffle = True)
            
            ##
            # load the associated model weights
            ##
            self.learner.load(iteration_to_weights[current_iteration])
            
            ##
            # make predictions for the current validation set
            ##
            self.predict_on_tiles(prediction_type=prediction_type, 
                                  dataset_type=shared.enums.DatasetType.validation,
                                  tile_size=tile_size, 
                                  batch_size=batch_size)