import fastai2
from fastai2.vision.all import *
import wsi_processing_pipeline
from wsi_processing_pipeline.tile_extraction import slide, filter, tiles, util


def label_ws_image(f):    
            name=f.classification_labels
                
            return name    

def wsi_image(tl:wsi_processing_pipeline.tile_extraction.tiles.Tile,                           
                          )->fastai2.vision.data.Image:
            '''This function is just for showcasing.
            It takes a tile object (wsi-preprocessing pipeline) and returns a WsiImage for displaying purposes '''
                       
            name=tl.classification_labels     
            img=WsiImage.create(tl)
            return WsiImage(img,name)

        
        
        
class WsiImage(Tuple):
    '''This class can 
    create a PIL Image from a Tile object.
    additionally the 
    show method enables to display the Tile object as an image'''
    @classmethod
    def create(cls, f):         
        wsi_path=f.wsi_path
        x=f.get_x()
        y=f.get_y()         
        width = f.get_width()
        height = f.get_height()
        level = f.level
        tile = tiles.ExtractTileFromWSI(path=wsi_path, x=x, y=y, width=width, height=height, level=level)        
        tile = TensorImage(image2tensor(tile))
        #tile=PILImage.create(tile)
        return tile
    
    def show(self, ctx=None, **kwargs):    
        img,title = self
        if not isinstance(img, Tensor):            
            t = tensor(img)
            t = t.permute(2,0,1)
        else: t = img
        #line = t.new_zeros(t.shape[0], t.shape[1], 10)
        #print(type(t))
        return show_image(t,title=title, ctx=ctx, **kwargs)
    
### custom Transforms ####    
class WsiTransform(Transform): 
    ''' A subclass of fastai Transform. Takes a TileObject and applies the image transform as taking tile object and returning 
    WsiImage aka PIL Image transformation.
    '''
    def __init__(self, tl:wsi_processing_pipeline.tile_extraction.tiles.Tile, splits):
        self.tl= tl
       
    def encodes(self,f):          
        
        return wsi_image(f)   

@typedispatch
def show_batch(x:WsiImage, y, samples, ctxs=None, max_n=6, nrows=None, ncols=2, figsize=None, **kwargs):     
    if figsize is None: figsize = (ncols*6, max_n//ncols * 3)
    if ctxs is None: ctxs = get_grid(min(x[0].shape[0], max_n), nrows=None, ncols=ncols, figsize=figsize)      
    for i,ctx in enumerate(ctxs): WsiImage(x[0][i], x[1][i]).show(ctx=ctx)
                                                                  

###TransformBlock###
def WsiImageBlock(): return TransformBlock(type_tfms=WsiImage.create, batch_tfms=IntToFloatTensor)

    
    
