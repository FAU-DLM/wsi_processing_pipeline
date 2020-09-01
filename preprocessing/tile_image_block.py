import fastai
from fastai.vision.all import *
import wsi_processing_pipeline
from wsi_processing_pipeline.tile_extraction import slide, filter, tiles, util
from wsi_processing_pipeline.preprocessing.name_getter import NameGetter
from wsi_processing_pipeline.preprocessing.objects import NamedObject
import shared


class TileImage():
    '''
    This class creates a TensorImage from a wsi_processing_pipeline.shared.tiles.Tile object.
    additionally the show method enables to display the Tile object as an image
    '''
    @classmethod
    def create(cls, t:shared.tile.Tile, **kwargs):
        if not isinstance (t, shared.tile.Tile):
            raise ValueError(f'{type(t)} not supported')
        return fastai.vision.core.PILImage.create(t.get_np_tile())
           
    def show(self, ctx=None, **kwargs):
        img,title = self
        if not isinstance(img, Tensor):            
            t = tensor(img)
            t = t.permute(2,0,1)
        else: t = img
        #line = t.new_zeros(t.shape[0], t.shape[1], 10)
        return fastai.torch_core.show_image(t,title=title, ctx=ctx, **kwargs)   

def label_tl_image(t:shared.tile.Tile): 
    return t.get_labels()
                    

def tile_image(t:shared.tile.Tile)->fastai.vision.data.Image:
    '''
    This function is just for showcasing.
    It takes a tile object (wsi-preprocessing pipeline) and returns a WsiImage for displaying purposes 
    '''    
    return TileImage(TileImage.create(t), t.get_labels())


class TileTransform(Transform): 
    ''' A subclass of fastai Transform. Takes a TileObject and applies the image transform as taking tile object and returning 
    WsiImage aka PIL Image transformation.
    '''
    def __init__(self, t:shared.tile.Tile, splits):
        self.t = t
       
    def encodes(self,t:shared.tile.Tile):
        return tile_image(t)     


@typedispatch
def show_batch(x:TileImage, y, samples, ctxs=None, max_n=6, nrows=None, ncols=2, figsize=None, **kwargs):  
    if figsize is None: figsize = (ncols*6, max_n//ncols * 3)
    if ctxs is None: ctxs = get_grid(min(x[0].shape[0], max_n), nrows=None, ncols=ncols, figsize=figsize)      
    for i,ctx in enumerate(ctxs): TileImage(x[0][i], x[1][i]).show(ctx=ctx)


def TileImageBlock(): 
    return TransformBlock(type_tfms=TileImage.create, batch_tfms=IntToFloatTensor)