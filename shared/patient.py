#https://stackoverflow.com/questions/46641078/how-to-avoid-circular-dependency-caused-by-type-hinting-of-pointer-attributes-in
from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from .case import Case
    from .patient_manager import PatientManager


from typing import List, Callable, Tuple
from shared.case import Case
from shared.patient_manager import PatientManager
from shared.enums import DatasetType

class Patient:    
    patient_id:str = None
    patient_manager:PatientManager = None
    cases:List[Case] = None
    dataset_type:DatasetType = None
        
    def __init__(self, patient_id:str, patient_manager:PatientManager):
        self.cases = []
        self.patient_id = patient_id
        self.patient_manager = patient_manager