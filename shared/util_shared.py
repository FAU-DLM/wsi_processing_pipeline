#https://stackoverflow.com/questions/46641078/how-to-avoid-circular-dependency-caused-by-type-hinting-of-pointer-attributes-in
from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from .tile import Tile
    from .enums import DatasetType


import pickle

def save_as_pickle(obj:object, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)
    
  
def get_x(tile:Tile):
    return tile
def get_y(tile):
    return tile.get_labels()
def split(tile):
    return tile.get_dataset_type() == DatasetType.validation


from .tile import Tile
from .enums import DatasetType