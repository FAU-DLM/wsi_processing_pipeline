#https://stackoverflow.com/questions/46641078/how-to-avoid-circular-dependency-caused-by-type-hinting-of-pointer-attributes-in
from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from .patient import Patient
    from .wsi import WholeSlideImage

import numpy
import numpy as np
from typing import List, Callable, Tuple


class Case:  
    case_id:str = None
    patient:Patient = None
    whole_slide_images:List[WholeSlideImage] = None
    predictions_raw:Dict[str, float] = None # key: class name; value: tiles with that class / all tiles
    predictions_thresh:Dict[str, bool] = None # key: class name; value: bool
            
    def __init__(self, case_id:str, patient:Patient):
        self.whole_slide_images = []
        self.case_id = case_id
        self.patient = patient
        
    def get_tiles(self)-> List[shared.tile.Tile]:
        tls = []
        for wsi in self.whole_slide_images:
            for roi in wsi.regions_of_interest:
                for tile in roi.tiles:
                    tls.append(tile)
        return tls
    
    def get_labels(self)-> List[Union[str, int]]:
        """
        iterates over all its associated tiles and returns a list of all found classes
        """
        labels = []
        for t in self.get_tiles():
            for l in t.labels:
                labels.append(l)
        return list(set(labels))
    
    def get_predictions_one_hot_encoded(self)->numpy.ndarray:
        """
            Returns:
                numpy array with one hot encoded labels
        """
        return np.array(list(self.predictions_thresh.values())).astype(np.int0)
    
    def get_labels_one_hot_encoded(self, vocab = None)->numpy.ndarray:
        """
            Arguments:
                vocab: A list of the used classes. It is very important that the order is the same as in 
                        learner.dls.vocab. The default value only works, if predict_on_tiles() has already been called.
            Returns:
                numpy array with one hot encoded labels
        """
        if(vocab == None and self.predictions_raw == None):
            raise ValueError('Either specify a vocab (e.g. learner.dls.vocab) or call Predictor.predict_on_tiles())')
        elif(vocab == None):
            vocab = self.predictions_raw.keys()
        
        labels_one_hot_encoded = np.zeros(len(vocab))
        labels_list = self.get_labels()
        for n, Class in enumerate(vocab):
            if(Class in labels_list):
                labels_one_hot_encoded[n] = 1
        return labels_one_hot_encoded.astype(np.int0)