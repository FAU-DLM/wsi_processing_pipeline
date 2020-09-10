from enum import Enum
class DatasetType(Enum):
    train = 0
    validation = 1
    test = 2
    all = 3 
    
    
class TissueQuantity(Enum):
  NONE = 0
  LOW = 1
  MEDIUM = 2
  HIGH = 3
    
class PredictionType(Enum):
    preextracted_tiles = 0
    tiles_on_the_fly = 1
    
class DataframeLevel(Enum):
    case_level = 0
    slide_level = 1
    
class EvaluationLevel(Enum):
    tile = 0
    slide = 1
    case = 2
    
class GradCamResult(Enum):
    predicted = 0
    targets = 1