#https://stackoverflow.com/questions/46641078/how-to-avoid-circular-dependency-caused-by-type-hinting-of-pointer-attributes-in
from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from .patient import Patient

from typing import List, Callable, Tuple
import pathlib
from pathlib import Path
Path.ls = lambda x: [p for p in list(x.iterdir()) if '.ipynb_checkpoints' not in p.name]
import functools
from sklearn.model_selection import StratifiedKFold, KFold
import shared
from shared.enums import DatasetType
from shared import roi

##
# rest of the imports from shared is at the end of the document to solve circular dependency problem
# https://stackoverflow.com/questions/894864/circular-dependency-in-python
##


class PatientManager:
    
    patients:List[Patient] = None
    
    def __init__(self):
        self.patients = []
        
    def create_from_tilesummaries(self, 
                                  tilesummaries:List[tile_extraction.tiles.TileSummary], 
                                  patient_id_getter:Callable, 
                                  case_id_getter:Callable, 
                                  slide_id_getter:Callable, 
                                  labels_getter:Callable):
        """
        Inits self.patients from tilesummaries. 
        On how to create tilesummaries: 
            https://github.com/FAU-DLM/wsi_processing_pipeline/blob/master/tile_extraction/example.ipynb
            
        Arguments:
            tilesummaries:
            patient_id_getter: method that maps from the tilesummary's whole-slide image path to a patient id
            case_id_getter: method that maps from the tilesummary's whole-slide image path to a case id
            slide_id_getter: method that maps from the tilesummary's whole-slide image path to a slide id
            labels_getter: method that maps from the tilesummary's whole-slide image path and a RegionOfInterest
                            to a list ob labels
        """
        
        # empty the lists, in case this function is called multiple times
        self.patients = []
        for ts in tilesummaries:
            for roi in ts.rois:
                roi.tiles = []
        
        #key: patient_id; value: Patient object
        patient_id_to_patient = {}
        for tilesummary in tilesummaries:
            ###
            # patient
            ###
            current_patient = None
            patient_id = patient_id_getter(tilesummary.wsi_path)
            if(patient_id not in patient_id_to_patient.keys()):
                current_patient = Patient(patient_id=patient_id, patient_manager=self)
                patient_id_to_patient[patient_id] = current_patient
                self.patients.append(current_patient)
            else:
                current_patient = patient_id_to_patient[patient_id]
            
            ###
            # case
            ###
            case_id = case_id_getter(tilesummary.wsi_path)
            current_case = None
            for case in current_patient.cases:
                if(case.case_id == case_id):
                    current_case = case
                    break;
            if(current_case == None):
                current_case = Case(case_id=case_id, patient=current_patient)
                current_patient.cases.append(current_case)
            
            ###
            # whole-slide
            ###
            slide_id = slide_id_getter(tilesummary.wsi_path)
            current_slide = WholeSlideImage(slide_id=slide_id, case=current_case, path=tilesummary.wsi_path)
            current_case.whole_slide_images.append(current_slide)
            
            ###
            # regions of interest
            ###
            rois = tilesummary.rois
            
            assert (rois != None and len(rois) > 0)
            for roi in rois:
                current_slide.regions_of_interest.append(roi)
                roi.whole_slide_image = current_slide
                roi.labels = labels_getter(tilesummary.wsi_path, roi)
                for tile in tilesummary.top_tiles():               
                    if(tile.roi.roi_id == roi.roi_id):
                        roi.tiles.append(tile)
                        tile.labels = roi.labels
            
    
    def create_from_whole_slide_images(self, 
                                       paths:List[pathlib.Path],
                                        patient_id_getter:Callable, 
                                        case_id_getter:Callable, 
                                        slide_id_getter:Callable, 
                                        labels_getter:Callable, 
                                       regions_of_interest:RegionOfInterestDefinedByCoordinates):
        """
        Inits self.patients from whole-slide image paths. 
        Convenvience function, that will do the tile extraction process for you with some default
        parameters.
        
        Arguments:
            paths: paths to the whole-slide images
            patient_id_getter: method that maps from the whole-slide image path 
                                and RegionOfInterestDefinedByCoordinates object to a patient id
            case_id_getter: method that maps from the whole-slide image path
                            and RegionOfInterestDefinedByCoordinates object to a case id
            slide_id_getter: method that maps from the whole-slide image path
                             and RegionOfInterestDefinedByCoordinates object to a slide id
        """
        raise NotImplementedError()
    
    def create_from_preextracted_regions_of_interest(self, 
                                       paths:List[pathlib.Path],
                                        patient_id_getter:Callable, 
                                        case_id_getter:Callable, 
                                        slide_id_getter:Callable, 
                                        labels_getter:Callable):
        """
        Inits self.patients from paths to already extracted regions of interest.
            
        Arguments:
            paths: paths to the regions of interest
            patient_id_getter: method that maps from the roi path to a patient id
            case_id_getter: method that maps from the roi path to a case id
            slide_id_getter: method that maps from the roi path to a slide id
        """
        raise NotImplementedError()
    
    def create_from_preextracted_tiles(self, 
                                       tile_paths:List[pathlib.Path],
                                        patient_id_getter:Callable, 
                                        case_id_getter:Callable, 
                                        slide_id_getter:Callable, 
                                        labels_getter:Callable):
        """
        Inits self.patients from paths to already extracted tiles.
            
        Arguments:
            paths: paths to the tiles
            patient_id_getter: method that maps from the tile image path to a patient id
            case_id_getter: method that maps from the tile image path to a case id
            slide_id_getter: method that maps from the tile image path to a slide id
            labels_getter: function that maps from a tile_path to the tile's labels
        """
        
        # empty the list, in case this function is called multiple times
        self.patients = []
        
        #key: patient_id; value: Patient object
        patient_id_to_patient = {}
        for tile_path in tile_paths:
            ###
            # patient
            ###
            current_patient = None
            patient_id = patient_id_getter(tile_path)
            if(patient_id not in patient_id_to_patient.keys()):
                current_patient = Patient(patient_id=patient_id, patient_manager=self)
                patient_id_to_patient[patient_id] = current_patient
                self.patients.append(current_patient)
            else:
                current_patient = patient_id_to_patient[patient_id]
            
            ###
            # case
            ###
            case_id = case_id_getter(tile_path)
            current_case = None
            for case in current_patient.cases:
                if(case.case_id == case_id):
                    current_case = case
                    break;
            if(current_case == None):
                current_case = Case(case_id=case_id, patient=current_patient)
                current_patient.cases.append(current_case)
            
            ###
            # whole-slide
            ###
            slide_id = slide_id_getter(tile_path)
            current_slide = None
            for slide in current_case.whole_slide_images:
                if(slide.slide_id == slide_id):
                    current_slide = slide
                    break;
            if(current_slide == None):
                current_slide = WholeSlideImage(slide_id=slide_id, case=current_case)
                current_case.whole_slide_images.append(current_slide)
            
            ###
            # region of interest; just one dummy roi per wsi
            ###
            current_roi = None
            if(len(current_slide.regions_of_interest) == 0):
                current_roi = roi.RegionOfInterestDummy(slide_id, current_slide)
                current_slide.regions_of_interest.append(current_roi)
            else:
                current_roi = current_slide.regions_of_interest[0]
            tile = Tile(roi=current_roi, tile_path=tile_path, labels=labels_getter(tile_path))
            current_roi.tiles.append(tile)
            
            

    def split(self, splitter:Callable):
        """
        Convenience function that splits the patients into a train and a validation set and sets the dataset_type attribute of every 
        patient in self.patients using the split provided by "splitter".
        
        Arguments:
            splitter: a Callable, that takes the following input parameters (e.g. sklearn.model_selection.train_test_split):
                                    set of patient ids

                                    and returns three lists:
                                       list of patient ids for the training set
                                       list of patient ids for the validation set
                                       list of patient ids for the test set
        """

        patient_ids = [patient.patient_id for patient in self.patients]
        
        # sorting ensures a reproduceable split of the ids. If the same ids are given in a different order to the method,
        # without sorting it would result in a different split, even if random_state is the the same (and numpy.random.seed()).
        patient_ids.sort()
        
        ids_train, ids_val, ids_test = splitter(list(set(patient_ids)))              
        for patient in self.patients:
            if(patient.patient_id in ids_val):
                patient.dataset_type = DatasetType.validation
            elif(patient.patient_id in ids_train):
                patient.dataset_type = DatasetType.train
            elif(patient.patient_id in ids_test):
                patient.dataset_type = DatasetType.test
            else:
                assert False
                
                
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
        ids_test = []
        return ids_train, ids_val, ids_test 

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

    
    def __get_tiles(self, dataset_type:shared.enums.DatasetType):
        tls = []
        for patient in self.patients:
            for case in patient.cases:
                for wsi in case.whole_slide_images:
                    for roi in wsi.regions_of_interest:
                        for tile in roi.tiles:
                            if(dataset_type == shared.enums.DatasetType.all or tile.get_dataset_type() == dataset_type):
                                tls.append(tile)
                            
        return tls
    
    def get_all_tiles(self)->List[shared.tile.Tile]:
        """
            Convenience function that gets all tiles.
        """
        return self.__get_tiles(dataset_type = shared.enums.DatasetType.all)
    
    def get_tiles(self, dataset_type:shared.enums.DatasetType)->List[shared.tile.Tile]:
        return self.__get_tiles(dataset_type = dataset_type)

    
    def get_patients(self, dataset_type:shared.enums.DatasetType)->List[shared.patient.Patient]:
        return [p for p in self.patients if(dataset_type == shared.enums.DatasetType.all or p.dataset_type == dataset_type)]
                    
from .patient import Patient
from .case import Case
from .wsi import WholeSlideImage
from .tile import Tile