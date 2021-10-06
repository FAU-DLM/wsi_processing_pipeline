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
#from shared.enums import DatasetType, EvaluationLevel
from shared import roi
import numpy as np
import sklearn
import random
from tqdm import tqdm
import multiprocessing
from hashlib import sha256

##
# rest of the imports from shared is at the end of the document to solve circular dependency problem
# https://stackoverflow.com/questions/894864/circular-dependency-in-python
##



#need to be on the top level of the module and not an instance method to be pickled in multiprocessing pool
#https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function
def remove_object(obj:Union[shared.tile.Tile, shared.wsi.WholeSlideImage, shared.case.Case]):
    if(type(obj) != shared.tile.Tile):
        print(type(obj))
    obj.set_removed_flag(value=True)
    
def remove_objects(objs:List[Union[shared.tile.Tile, shared.wsi.WholeSlideImage, shared.case.Case]], 
                   verbose=False):    
        if(verbose):
            pbar = tqdm(total=len(objs))
        
        def update():
            if(verbose):
                pbar.update()
            
        def error(e):
            print(e)
        
        #with multiprocessing.Pool() as pool:
        #    for o in objs:
        #        pool.apply_async(remove_object, 
        #                         kwds={"obj":o}, 
        #                               callback=update, 
        #                               error_callback=error)                    
        #    pool.close()
        #    pool.join()  
                
        for obj in objs:
            remove_object(obj=obj)
            update()


            
            
class PatientManager:
        
    __patients:List[Patient] = None
    
    def __init__(self):
        self.__patients = []

    def __get_patients(self)->List[Patient]:
        return [p for p in self.__patients if(not p.is_removed())]
    
    def add_patient(self, patient:Patient):
        self.__patients.append(patient)
############################################# init methods ###################################################################        

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
        self.__patients = []
        __rois = []
        #key: patient_id; value: Patient object
        patient_id_to_patient = {}
        for tilesummary in tqdm(tilesummaries):
            
            # skip tilesummaries with 0 tiles
            if(len(tilesummary.top_tiles()) == 0):
                continue
            
            
            ###
            # patient
            ###
            current_patient = None
            patient_id = patient_id_getter(tilesummary.wsi_path)
            if(patient_id not in patient_id_to_patient.keys()):
                current_patient = Patient(patient_id=patient_id, patient_manager=self)
                patient_id_to_patient[patient_id] = current_patient
                self.add_patient(current_patient)
            else:
                current_patient = patient_id_to_patient[patient_id]
            
            ###
            # case
            ###
            case_id = case_id_getter(tilesummary.wsi_path)
            current_case = None
            for case in current_patient.get_cases():
                if(case.case_id == case_id):
                    current_case = case
                    break;
            if(current_case == None):
                current_case = Case(case_id=case_id, patient=current_patient)
                current_patient.add_case(current_case)
            
            ###
            # whole-slide
            ###
            slide_id = slide_id_getter(tilesummary.wsi_path)
            current_slide = WholeSlideImage(slide_id=slide_id, case=current_case, path=tilesummary.wsi_path)
            current_case.add_whole_slide_image(current_slide)
            
            ###
            # regions of interest
            ###
            rois = tilesummary.rois
            
            assert (rois != None and len(rois) > 0)
            for roi in rois:
                
                __rois.append(roi)
                
                roi.reset_tiles() #reset tiles in case this function is called multiple times
                current_slide.add_region_of_interest(roi)
                roi.whole_slide_image = current_slide
                roi.labels = labels_getter(tilesummary.wsi_path, roi)
                for tile in tilesummary.top_tiles():               
                    if(tile.roi.roi_id == roi.roi_id):
                        tile.set_removed_flag(value=False) # this flag might still be True from a previous call
                        roi.add_tile(tile)
                        tile.labels = roi.labels
                        
        #__rois = self.get_rois(dataset_type=shared.enums.DatasetType.all)
        roi_ids = [r.roi_id for r in __rois]
        if(len(roi_ids) != len(set(roi_ids))):
            raise ValueError('The rois do not have unique ids.')
            
    
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
        for tile_path in tqdm(tile_paths):
            ###
            # patient
            ###
            current_patient = None
            patient_id = patient_id_getter(tile_path)
            if(patient_id not in patient_id_to_patient.keys()):
                current_patient = Patient(patient_id=patient_id, patient_manager=self)
                patient_id_to_patient[patient_id] = current_patient
                self.add_patient(current_patient)
            else:
                current_patient = patient_id_to_patient[patient_id]
            
            ###
            # case
            ###
            case_id = case_id_getter(tile_path)
            current_case = None
            for case in current_patient.get_cases():
                if(case.case_id == case_id):
                    current_case = case
                    break;
            if(current_case == None):
                current_case = Case(case_id=case_id, patient=current_patient)
                current_patient.add_case(current_case)
            
            ###
            # whole-slide
            ###
            slide_id = slide_id_getter(tile_path)
            current_slide = None
            for slide in current_case.get_whole_slide_images():
                if(slide.slide_id == slide_id):
                    current_slide = slide
                    break;
            if(current_slide == None):
                current_slide = WholeSlideImage(slide_id=slide_id, case=current_case)
                current_case.add_whole_slide_image(current_slide)
            
            ###
            # region of interest; just one dummy roi per wsi
            ###
            current_roi = None
            if(len(current_slide.get_regions_of_interest()) == 0):
                current_roi = roi.RegionOfInterestDummy(slide_id, current_slide)
                current_slide.add_region_of_interest(current_roi)
            else:
                current_roi = current_slide.get_regions_of_interest()[0]
            tile = Tile(roi=current_roi, tile_path=tile_path, labels=labels_getter(tile_path))
            current_roi.add_tile(tile)
            
############################################# END init methods ###################################################################     


############################################# split methods ###################################################################### 

    def split_by_function(self, splitter:Callable, random_state:int):
        """
        Splits the patients into a train, validation and test set and sets the dataset_type attribute of every 
        patient in self.patients using the split provided by "splitter".
        
        Arguments:
            splitter: a Callable, that takes the following input parameters:
                                    set of patient ids

                                    and returns three lists:
                                       list of patient ids for the training set
                                       list of patient ids for the validation set
                                       list of patient ids for the test set
        """

        np.random.seed(random_state)
        random.seed(random_state)
        
        # sorting ensures a reproduceable split of the ids. 
        # If the same ids are given in a different order to the method,
        # without sorting it would result in a different split, 
        # even if random_state is the the same (and numpy.random.seed()).
        patient_ids = sorted(list(set([patient.patient_id for patient in self.__get_patients()])))
                
        ids_train, ids_val, ids_test = splitter(patient_ids)              
        for patient in self.__get_patients():
            if(patient.patient_id in ids_val):
                patient.dataset_type = shared.enums.DatasetType.validation
            elif(patient.patient_id in ids_train):
                patient.dataset_type = shared.enums.DatasetType.train
            elif(patient.patient_id in ids_test):
                patient.dataset_type = shared.enums.DatasetType.test
            else:
                assert False
    
    
    def __split(self, patient_ids:List[str], train_size:float, validation_size:float, test_size:float, random_state:int):
        """
        Convencience function that splits the patients randomly into a train, validation and test set 
        according to the given sizes.
        
        Arguments:
            train_size: in range (0,1]
            validation_size: [0, 1)
            test_size: [0,1]
            random_seed: 
        """
        # checks
        if(train_size <= 0 or train_size > 1):
            raise ValueError("train_size must be in range (0,1]")        
        if(validation_size < 0 or validation_size >= 1):
            raise ValueError("validation_size must be in range [0, 1)")
        if(test_size < 0 or test_size > 1):
            raise ValueError("test_size must be in range [0,1]")
        if(train_size + validation_size + test_size != 1):
            raise ValueError("train_size, validation_size and test_size must add up to 1")
        
        np.random.seed(random_state)
        random.seed(random_state)
        
        # sorting ensures a reproduceable split of the ids. 
        # If the same ids are given in a different order to the method,
        # without sorting it would result in a different split, 
        # even if random_state is the the same (and numpy.random.seed()).
        patient_ids.sort()
        
        # edge cases
        if(validation_size <= 0 and test_size <= 0):
            return patient_ids, [], []
        if(test_size <= 0):
            train_ids, val_ids = sklearn.model_selection.train_test_split(patient_ids, 
                                                            train_size=train_size, 
                                                            #test_size=validation_size, #not necessary
                                                            random_state=random_state, 
                                                            shuffle=True, 
                                                            stratify=None)
            return train_ids, val_ids, []
        
        else:
            train_and_val_ids, test_ids = sklearn.model_selection.train_test_split(patient_ids, 
                                                            train_size=train_size+validation_size, 
                                                            test_size=test_size, 
                                                            random_state=random_state, 
                                                            shuffle=True, 
                                                            stratify=None)
            #update train_size to get the train_size of all patient_ids
            train_size = train_size/(1-test_size)
            train_ids, val_ids = sklearn.model_selection.train_test_split(train_and_val_ids, 
                                                            train_size=train_size, 
                                                            #test_size=1-train_size, #not necessary
                                                            random_state=random_state, 
                                                            shuffle=True, 
                                                            stratify=None)
            return train_ids, val_ids, test_ids
        
    
    
    def split(self, train_size:float, validation_size:float, test_size:float, random_state:int):
        """
        Convencience function that splits the patients randomly into a train, validation and test set 
        according to the given sizes.
        
        Arguments:
            train_size: in range (0,1]
            validation_size: [0, 1)
            test_size: [0,1]
            random_state: a random seed that can be set to get consistent splits 
        """
        np.random.seed(random_state)
        random.seed(random_state)
        
        splitter = functools.partial(self.__split, 
                                     train_size=train_size, 
                                     validation_size=validation_size,
                                     test_size=test_size,
                                     random_state=random_state)
        self.split_by_function(splitter, random_state)
    
                
    def __split_KFold_cross_validation(self, 
                                       patient_ids:List[str],
                                       n_splits:int,
                                       current_iteration:int,
                                       random_state:int,
                                       shuffle:bool)->List[List[str]]:
        np.random.seed(random_state)
        random.seed(random_state)
        
        # sorting ensures a reproduceable split of the ids. 
        #If the same ids are given in a different order to the method,
        # without sorting it would result in a different split, 
        # even if random_state is the the same (and numpy.random.seed()).
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
        
        np.random.seed(random_state)
        random.seed(random_state)
        
        if current_iteration < 0 or current_iteration >= n_splits:
            raise ValueError(f'current_iteration must be in [0, {n_splits-1}]  (between 0 and n_splits-1)')
            
        splitter = functools.partial(self.__split_KFold_cross_validation, 
                                     n_splits=n_splits, 
                                     current_iteration=current_iteration, 
                                     random_state=random_state, 
                                     shuffle=shuffle)
        self.split_by_function(splitter, random_state)

        
############################################# END split methods ###################################################################        
        
############################################# getter methods ######################################################################
    
    def get_patients(self, dataset_type:shared.enums.DatasetType)->List[shared.patient.Patient]:
        #print(dataset_type is shared.enums.DatasetType.all)
        return [p for p in self.__get_patients() if(dataset_type == shared.enums.DatasetType.all \
                                                    or p.dataset_type==dataset_type)]
    
    
    def __get_objects_according_to_evaluation_level(self, 
                                                    level:shared.enums.EvaluationLevel, 
                                                    dataset_type:shared.enums.DatasetType) \
                                                    ->List[Union[shared.tile.Tile, \
                                                                 shared.wsi.WholeSlideImage, \
                                                                 shared.case.Case]]:
        objs = None
        
        if(level == shared.enums.EvaluationLevel.tile or str(level) == 'EvaluationLevel.tile'):
            objs = self.get_tiles(dataset_type = dataset_type)
        elif(level == shared.enums.EvaluationLevel.slide):
            objs = self.get_wsis(dataset_type=dataset_type)
        elif(level == shared.enums.EvaluationLevel.case):
            objs = self.get_cases(dataset_type=dataset_type)
        else:
            raise ValueError('Wrong value for level.')
            
        return objs
    
    
    def __get_tiles(self, dataset_type:shared.enums.DatasetType):
        tls = []
        for patient in self.get_patients(dataset_type=dataset_type):
            for case in patient.get_cases():
                for wsi in case.get_whole_slide_images():
                    for roi in wsi.get_regions_of_interest():
                        for tile in roi.get_tiles():
                            tls.append(tile)
                            
        return tls
    
    def get_all_tiles(self)->List[shared.tile.Tile]:
        """
            Convenience function that gets all tiles.
        """
        return self.__get_tiles(dataset_type = shared.enums.DatasetType.all)
    
    def get_tiles(self, dataset_type:shared.enums.DatasetType)->List[shared.tile.Tile]:
        return self.__get_tiles(dataset_type = dataset_type)
    
    def get_rois(self, dataset_type:shared.enums.DatasetType)->List[shared.wsi.WholeSlideImage]:
        rois = []
        for patient in self.get_patients(dataset_type=dataset_type):
            for case in patient.get_cases():
                for wsi in case.get_whole_slide_images():
                    for r in wsi.get_regions_of_interest():
                        rois.append(r)
                            
        return rois
        
    def get_wsis(self, dataset_type:shared.enums.DatasetType)->List[shared.wsi.WholeSlideImage]:
        wsis = []
        for patient in self.get_patients(dataset_type=dataset_type):
            for case in patient.get_cases():
                for wsi in case.get_whole_slide_images():
                    wsis.append(wsi)
                            
        return wsis
    
    def get_cases(self, dataset_type:shared.enums.DatasetType)->List[shared.case.Case]:
        cases = []
        for patient in self.get_patients(dataset_type=dataset_type):
            for case in patient.get_cases():
                cases.append(case)
                            
        return cases
      
    def get_classes(self)->List[str]:
        classes = set()
        for t in self.get_all_tiles():
            for l in t.get_labels():
                classes.add(l)
        return sorted(list(classes))
    
    def get_class_distribution(self, 
                               level:shared.enums.EvaluationLevel, 
                               dataset_type:shared.enums.DatasetType)->(Dict[str, int], Dict[str, float]):
        """
        Arguments:
            level: 
            dataset_type:
        Returns:
            total amount of cases/slides/tiles (depending on level) in the dataset (depending on dataset_type) 
            and two dictionaries with the class names as keys:
                1st: values == absolute numbers of cases/slides/tiles (depending on level) with that class as label 
                2nd: values == percentage of cases/slides/tiles (depending on level) with that class as label 
        """        
        classes = self.get_classes()
        dict_class_to_n = {}
        for c in classes:
            dict_class_to_n[c] = 0
        
        objs = self.__get_objects_according_to_evaluation_level(level=level, dataset_type=dataset_type)
        for obj in objs:
            for l in obj.get_labels():
                dict_class_to_n[l] +=1
        
        dict_class_to_percentage = {}
        for Class, n in dict_class_to_n.items():
            dict_class_to_percentage[Class] = dict_class_to_n[Class]/len(objs)
        
        return len(objs), dict_class_to_n, dict_class_to_percentage

############################################# END getter methods ###################################################################
    
    
############################################# remove methods ###################################################################    
       
    def remove_cases(self, cases:List[shared.case.Case], verbose=False)-> Dict[shared.case.Case, bool]:
        """
        Sets the "removed" flags of the given cases to True.
        """
        remove_objects(objs=cases, verbose=verbose)
    

    def remove_slides(self, wsis:List[shared.wsi.WholeSlideImage], verbose=False)\
                        -> Dict[shared.wsi.WholeSlideImage, bool]:
        """
        Sets the "removed" flags of the given wsis to True.
        """
        remove_objects(objs=wsis, verbose=verbose)
    
    def remove_tiles(self, tiles:List[shared.tile.Tile], verbose=False)-> Dict[shared.tile.Tile, bool]:
        """
        Sets the "removed" flags of the given tiles to True.
        """
        remove_objects(objs=tiles, verbose=verbose)
    

############################################# END remove methods ###################################################################
    

############################################# downsampling and reduce methods ######################################################   
    
    def undersample(self, 
                   level:shared.enums.EvaluationLevel = shared.enums.EvaluationLevel.tile, 
                   dataset_type:shared.enums.DatasetType = shared.enums.DatasetType.train, 
                   delta:float=0.03, 
                   verbose=False, 
                   minimum_number_of_tiles:int = 50):
        """
        Removes cases/wsis/tiles (depending on the specified level) until the share of the most present class
        and the share of the least present class only differ by the specified delta in the specified dataset type.
        
        Arguments:
            level:
                    case: removes whole cases
                    slide: removes complete whole-slide images
                    tile: removes only tiles
            delta: the method stops removing cases/wsis/tiles as soon as the difference between the share of
                    most present class and the least present class is smaller or equal to delta
            verbose: if True some more info during the process is printed
            minimum_number_of_tiles: only relevant if level == shared.enums.EvaluationLevel.tile;
                                        a tile only gets removed, if there are at least <minimum_numer_of_tiles> tiles
                                        left of its corresponding whole-slide image
        
        """
        
        distr = self.get_class_distribution(level=level, dataset_type=dataset_type)
        most_present_class_name, most_present_class_percentage = max(distr[2].items(), key=lambda x : x[1])
        least_present_class_name, least_present_class_percentage = min(distr[2].items(), key=lambda x : x[1])
        
        
        count = 0
        while((most_present_class_percentage - least_present_class_percentage) > delta):
            distr = self.get_class_distribution(level=level, dataset_type=dataset_type)
            most_present_class_name, most_present_class_percentage = max(distr[2].items(), key=lambda x : x[1])
            least_present_class_name, least_present_class_percentage = min(distr[2].items(), key=lambda x : x[1])
        
            if(verbose and (count % 3 == 0)):
                print(distr[2])
            
            #sort to get a consistent downsample result
            objs = sorted(self.__get_objects_according_to_evaluation_level(level=level, dataset_type=dataset_type), 
                          key = lambda x : sha256(str(x).encode()).hexdigest())
            
                               
            
            objs_suitable_for_removal = [obj for obj in objs if(most_present_class_name in obj.get_labels() 
                                                                and least_present_class_name not in obj.get_labels())]
            
            # filter out all tiles, that shall not be removed, since there are only less than <minimum_number_of_tiles>
            # tiles left of their corresponding whole-slide image
            if(level == shared.enums.EvaluationLevel.tile):
                objs_suitable_for_removal = [obj for obj in objs_suitable_for_removal 
                                             if len(obj.roi.whole_slide_image.get_tiles()) >= minimum_number_of_tiles]
                
                # check if the amount of objs_suitable_for_removal is zero and return with a notice. 
                # This would result in an endless loop otherwise.
                if(len(objs_suitable_for_removal) == 0):
                    print(f"Could not reach the desired delta of {delta}.\
                            Only reached a delta of {most_present_class_percentage - least_present_class_percentage}\
                            Try a smaller <minimum_number_of_tiles>.")
                    return
            
            #remove a whole bunch at a time => faster
            n_to_remove = int((most_present_class_percentage - least_present_class_percentage)/30*len(objs))
            if(verbose):
                print(f'n to remove: {n_to_remove}')
                            
            remove_objects(objs=objs_suitable_for_removal[:n_to_remove], verbose=verbose)
            
            count += 1

            
    def reduce_tiles(self, 
                     dataset_type:shared.enums.DatasetType, 
                     remaining_percentage:float, 
                     random_seed:int, 
                     verbose=False)->Dict[object, bool]:
        """
        This method reduces the total number of tiles of the given dataset_type to (1 - remaining_percentage) percent.
        Arguments:
            dataset_type:
            remaining_percentage: value in range (0.0;1.0); percentage of tiles that will remain of each WSI
            random_seed: int number, use the same number to get same results in every run
        """
        if(remaining_percentage <= 0.0 or remaining_percentage >= 1.0):
            raise ValueError("The parameter remaining percentage has to be in the range (0.0;1.0)")
            
        random.seed = random_seed  
                
        # sorted to get a consistent sample for the same random seed
        tiles = sorted(self.__get_tiles(dataset_type=dataset_type), key=lambda t: t.get_name())
        print(f'total number of tiles: {len(tiles)}')
        k_to_remove = int(len(tiles)*(1-remaining_percentage))
        print(f'number of tiles that will be removed: {k_to_remove}')
        tiles_to_remove = random.sample(population=tiles, k=k_to_remove)
        remove_objects(objs=tiles_to_remove, verbose=True)
            
            
############################################# END downsampling and reduce methods ####################################################          
                       
import patient    
import case
import wsi
import tile
from patient import Patient
from case import Case
from wsi import WholeSlideImage
from tile import Tile