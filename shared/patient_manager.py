#https://stackoverflow.com/questions/46641078/how-to-avoid-circular-dependency-caused-by-type-hinting-of-pointer-attributes-in
from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from .patient import Patient

from typing import List, Callable, Tuple
import pathlib
from pathlib import Path
Path.ls = lambda x: [p for p in list(x.iterdir()) if '.ipynb_checkpoints' not in p.name]

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
        
        #key: patient_id; value: Patient object
        patients = {}
        for tilesummary in tilesummaries:
            ###
            # patient
            ###
            current_patient = None
            patient_id = patient_id_getter(tilesummary.wsi_path)
            if(patient_id not in patients.keys()):
                current_patient = Patient(patient_id=patient_id, PatientManager=self)
                patients.append(current_patient)
            else:
                current_patient = patients[patient_id]
            
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
            current_slide = WholeSlideImage(slide_id=slide_id, case=current_case)
            current_case.whole_slide_images.append(current_slide)
            
            ###
            # regions of interest
            ###
            rois = tilesummary.rois
            assert (rois != None and len(rois) > 0)
            for roi in rois:
                roi.whole_slide_image = current_slide
                roi.labels = labels_getter(tilesummary.wsi_path, roi)
                for tile in tilesummary.top_tiles:
                    tile.labels = labels
                    if(tile.roi.roi_id == roi.roi_id):
                        roi.tiles.append(tile)
            
    
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
        pass
    
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
        pass
    
    def create_from_preextracted_tiles(self, 
                                       paths:List[pathlib.Path],
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
        """
        #key: patient_id; value: Patient object
        patients = {}
        for tile_path in tile_paths:
            ###
            # patient
            ###
            current_patient = None
            patient_id = patient_id_getter(tile_path)
            if(patient_id not in patients.keys()):
                current_patient = Patient(patient_id=patient_id, PatientManager=self)
                patients.append(current_patient)
            else:
                current_patient = patients[patient_id]
            
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
                current_roi = RegionOfInterestDummy(slide_id, current_slide)
                current_slide.regions_of_interest.append(current_roi)
            else:
                current_roi = current_slide.regions_of_interest[0]
            current_roi.tiles.append(tile_path)