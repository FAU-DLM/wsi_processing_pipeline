from enum import Enum
class DatasetType(Enum):
    train = 0
    validation = 1
    test = 2
    
    
class TissueQuantity(Enum):
  NONE = 0
  LOW = 1
  MEDIUM = 2
  HIGH = 3
    
class PredictionType(Enum):
    preextracted_tiles = 0
    tiles_on_the_fly = 1