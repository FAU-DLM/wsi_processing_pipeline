from __future__ import annotations #https://stackoverflow.com/questions/33837918/type-hints-solve-circular-dependency


from shared.case import Case
import typing
from typing import Dict


class WholeSlideImage:
    case:Case = None
    regions_of_interest:List[RegionOfInterest] = None
    slide_id:str = None
    path:pathlib.Path = None
    predictions_raw:Dict[str, float] = None # key: class name; value: tiles with that class / all tiles
    predictions_thresh:Dict[str, bool] = None # key: class name; value: bool
      
    def __init__(self, slide_id:str, case:Case, path=None):
        self.case = case
        self.slide_id = slide_id
        self.path = path
        self.regions_of_interest = []
        
    def get_tiles(self)-> List[shared.tile.Tile]:
        tls = []
        for roi in self.regions_of_interest:
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