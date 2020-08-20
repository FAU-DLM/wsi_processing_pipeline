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
            
    def __init__(self, case_id:str, patient:Patient):
        self.whole_slide_images = []
        self.case_id = case_id
        self.patient = patient