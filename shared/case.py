#https://stackoverflow.com/questions/46641078/how-to-avoid-circular-dependency-caused-by-type-hinting-of-pointer-attributes-in
from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from .patient import Patient
    from .wsi import WholeSlideImage

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