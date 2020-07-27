import fastai2
import wsi_processing_pipeline
from wsi_processing_pipeline.preprocessing.objects import NamedObject
import pandas as pd

class Predictor(object):
    
    def __init__(self, 
                 learner:fastai2.learner.Learner,                  
                 path, 
                 tta:bool=False,
                 thresh:float=0.5, 
                 exclude_failed:bool=False, 
                 dl:fastai2.data.core.TfmdDL=None):
        
        if learner is None or not isinstance(learner, fastai2.learner.Learner):            
            raise AssertionError('please make sure to use a "fastai2.learner.Learner" as "learner" input') 
        
        if dl is not None and not isinstance(dl, fastai2.data.core.TfmdDL):        
            raise AssertionError('please make sure to use a "fastai2.data.core.TfmdDL" as dataloader - dl - input')
        
        elif dl is not None and isinstance(dl, fastai2.data.core.TfmdDL):
            self.dl = dl
            self.ds = dl.dataset
        
        elif dl is None:
            self.dl = learner.dls.valid
            self.ds = learner.dls.valid_ds
        else:
            assert False # This case should not happen
                    
        self.learner=learner        
        self.path=path
        self.tta=tta
        self.thresh=thresh
        self.exclude_failed=exclude_failed
                        
        self.cat = fastai2.data.transforms.Categorize(vocab=self.ds.vocab)
        self.ds_items_checker()

       
    def ds_items_checker(self):        
        if isinstance(self.ds.items, list):
           
            if any(isinstance(el, NamedObject) for el in self.ds.items) or any(isinstance(el, WsiOrRoiObject) for el in self.ds.items):
                self.ds_items=self.ds.items#ObjectManager(self.ds.items).objects  
                
            elif any(isinstance(el, tiles.Tile) for el in self.ds.items):
              
                self.ds_items=self.ds.items
               
            else:
                raise AssertionError('Items of a dataset should be of type list containing NamedObjects or TileObjects or of a pandas dataframe ')            
                     
        elif isinstance(self.ds.items, pd.core.frame.DataFrame ):
                self.ds_items=self.ds.items
                
        else:
            raise AssertionError('Items of a dataset should be of type list containing NamedObjects or TileObjects or of a pandas dataframe ')            
                     
        
    def tile_id_checker(self):   
        if self.tile_ids is None:
            pass
        
        elif type(self.tile_ids) is list:            
            pass

        elif type(self.tile_ids) is str:
            self.tile_ids=[self.tile_ids]
            
                      
        elif isinstance(self.tile_ids, pathlib.PosixPath):     
            self.tile_ids=[str(self.tile_ids)]
                          
        elif isinstance(self.tile_ids, NamedObject) or isinstance(self.tile_ids, WsiOrRoiObject):
            self.tile_ids=[self.tile_ids]
        elif isinstance(self.tile_ids, tiles.Tile):
            self.tile_ids=[self.tile_ids]   
        else:
            raise AssertionError('tile_ids should be a string/PosixPath or a list of strings/PosixPaths containing the Path to the image')
        
        self.match_ds_items_and_tile_ids()
    
      
    def match_ds_items_and_tile_ids(self):
        if type(self.tile_ids) is list: 
            if any(isinstance(el, NamedObject) for el in self.tile_ids) or any(isinstance(el, WsiOrRoiObject) for el in self.tile_ids):
                
                self.ds_items=[items for items in self.tile_ids if items in self.ds_items]
            elif any(isinstance(el, tiles.Tile) for el in self.tile_ids):
                self.ds_items=[items for items in self.tile_ids if items in self.ds_items]
                
            elif any(isinstance(el, str) for el in self.tile_ids):                               
                self.ds_items=self.ds_items[self.ds_items.fname.isin(self.tile_ids)]
                
            elif any(isinstance(el, pathlib.PosixPath) for el in self.tile_ids):                
                self.ds_items=self.ds_items[self.ds_items.fname.isin(self.tile_ids)]
            
            
                
        if self.tile_ids is None:
            pass  
   

    def get_prediction_per_tile(self, ids=None):
        
        self.tile_ids=ids        
        self.tile_id_checker()        
        
        cls=[]
        prob=[]        
        above_thr=[]        
        
        if isinstance(self.ds_items, pd.DataFrame):
           
            tile_ids=[]
            slide_ids=[]
            case_ids=[]
            labels=[]
            for ids, item in tqdm(self.ds_items.iterrows(),  total=self.ds_items.shape[0]):        
                    tile_id=item['fname']
                    tile_ids.append(tile_id)            
                    slide_ids.append(item['slide_id'])
                    case_ids.append(item['case_id'])
                    labels.append(item['labels'])
            true_cls=labels
            
        else:
            try:
                tile_ids=[obj.path for obj in self.ds_items]
            except:
                tile_ids=[obj.wsi_path for obj in self.ds_items]
            true_cls=[obj.classification_labels for obj in self.ds_items]
            case_ids=[obj.case_id for obj in self.ds_items]
            slide_ids=[obj.slide_id for obj in self.ds_items]
           
        if self.tile_ids is not None: 
            
            if isinstance(self.ds_items, pd.DataFrame):
            
                df=pd.DataFrame(data={'fname':tile_ids, 'labels':labels})           
                if len(df)>=9:
                    for i in range (1,12):                    
                        if len(df)%i == 0:                        
                            bs=i
                else: bs=len(df)        
                dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                                   get_x=ColReader('fname') , 
                                   get_y=ColReader('labels') ,   
                                   splitter=RandomSplitter(valid_pct=0),
                                  
                                   batch_tfms=[*aug_transforms(mult=1.0, 
                                                             do_flip=True, 
                                                             flip_vert=True, 
                                                             max_rotate=10.0, 
                                                             max_zoom=1.1, 
                                                             max_lighting=0.2, 
                                                             max_warp=0.2, 
                                                             p_affine=0.75, 
                                                             p_lighting=0.75, 
                                                             xtra_tfms=None, 
                                                             size=None, 
                                                             mode='bilinear', 
                                                             pad_mode='reflection', 
                                                             align_corners=True, 
                                                             batch=True, 
                                                             min_scale=1.0), 
                                             Normalize.from_stats(*imagenet_stats)
                                            
                                            ]
                                           )  
                self.dl=dblock.dataloaders(df, bs=bs)[0]
                
            else:
                if len(self.ds_items)>=9:
                    for i in range (1,12):                    
                        if len(self.ds_items)%i == 0:                        
                            bs=i
                else: bs=len(self.ds_items)
                t_dblock = DataBlock(
                    blocks=(TileImageBlock, CategoryBlock),                   
                    get_x=lambda x: x, 
                    get_y=lambda x: x.classification_labels,
                    splitter=RandomSplitter(valid_pct=0),
                    item_tfms=Resize(224),
                    batch_tfms=[*aug_transforms(mult=1.0, 
                                                             do_flip=True, 
                                                             flip_vert=True, 
                                                             max_rotate=10.0, 
                                                             max_zoom=1.1, 
                                                             max_lighting=0.2, 
                                                             max_warp=0.2, 
                                                             p_affine=0.75, 
                                                             p_lighting=0.75, 
                                                             xtra_tfms=None, 
                                                             size=None, 
                                                             mode='bilinear', 
                                                             pad_mode='reflection', 
                                                             align_corners=True, 
                                                             batch=True, 
                                                             min_scale=1.0), 
                                             Normalize.from_stats(*imagenet_stats)                                            
                                            ]
                    )
            
                self.dl=t_dblock.dataloaders(self.ds_items, bs=bs)[0]
                try:
                    tile_ids=[obj.path for obj in self.ds_items]
                except:
                    tile_ids=[obj.wsi_path for obj in self.ds_items]
                true_cls=[obj.classification_labels for obj in self.ds_items]
                case_ids=[obj.case_id for obj in self.ds_items]
                slide_ids=[obj.slide_id for obj in self.ds_items]
        
        
        
        if self.tta is False:
            prbs,_=self.learner.get_preds(dl=self.dl)

        else:            
            
            prbs, _=self.learner.tta(dl=self.dl, use_max=False)  
        
        
        for prb in prbs:
            prb=prb.numpy()
            
            cl=np.argmax(prb)
            prob.append(prb)
            
            cl = self.cat.decodes(cl)
            cls.append(cl)
            above_thresh=False

            if prb[np.argmax(prb)] > self.thresh:
                above_thresh=True
            above_thr.append(above_thresh)    
        
        tile_dataframe=pd.DataFrame(data={'tile_id':tile_ids,
                                     'slide_id':slide_ids,
                                     'case_id':case_ids,
                                     'predicted_class':cls,
                                     'true_class':true_cls,    
                                     'probabilities':prob,
                                     'above_threshold': above_thr,
                                     }
                               )
        
        if not isinstance(self.ds_items, pd.DataFrame):
            for obj,cl,pr,ab in zip(self.ds_items,cls,prob,above_thr):
                obj.predicted_class=cl
                obj.probabilities=pr
                obj.above_threshold=ab
            
        return tile_dataframe 

    
    def get_prediction_per_slide(self, ids=None):
        
        slide_ids=ids
        
        if slide_ids:                
            if type(slide_ids) is not list:
                try: 
                    slide_ids=[slide_ids]
                except:     
                    raise NotImplementedError
            
            ids=[] 
            
            if isinstance(self.ds_items, pd.DataFrame):
                for iid,item in zip(self.ds_items['slide_id'].to_list(),self.ds_items['fname'].to_list()):
                    
                    if iid in slide_ids:
                        ids.append(item)
                        
            if isinstance(self.ds_items, list):
                for objects in self.ds_items:
                    
                    if objects.slide_id in slide_ids:
                        ids.append(objects)
                    
                        
                
            if ids==[]:
                raise ValueError('None of your list of slide ids is contained within the provided dataset; please check your data!')
                

        
        tile_dataframe=self.get_prediction_per_tile(ids=ids)
        
        
        df_list=[]
        for unique_item in tile_dataframe['slide_id'].unique():
            
            classlabel=[]
            ass_prob=[]
            ids=[]
            for dat, thr, iid,true_cls, case_id in zip(tile_dataframe.loc[tile_dataframe['slide_id'] == unique_item, 'probabilities'],
                                tile_dataframe.loc[tile_dataframe['slide_id'] == unique_item, 'above_threshold'],
                                tile_dataframe.loc[tile_dataframe['slide_id'] == unique_item, 'tile_id'],
                                tile_dataframe.loc[tile_dataframe['slide_id'] == unique_item, 'true_class'],
                                tile_dataframe.loc[tile_dataframe['slide_id'] == unique_item, 'case_id']):        
                
                if self.exclude_failed:
                    if thr > self.thresh:

                        classlabel.append(self.cat.decodes(np.argmax(dat)))
                        ass_prob.append(dat[np.argmax(dat)])
                else:
                    classlabel.append(self.cat.decodes(np.argmax(dat)))
                    ass_prob.append(dat[np.argmax(dat)])

                ids.append(iid)

            cls_slide_lbl_counts=pd.DataFrame({'classlabel':classlabel})['classlabel'].value_counts().to_list()
            cls_slide_lbl=pd.DataFrame({'classlabel':classlabel})['classlabel'].value_counts().idxmax()

            idxmin=pd.DataFrame({'ass_prob':ass_prob})['ass_prob'].idxmin()
            idxmax=pd.DataFrame({'ass_prob':ass_prob})['ass_prob'].idxmax()
            min_slide_prob=pd.DataFrame({'ass_prob':ass_prob})['ass_prob'][idxmin]
            max_slide_prob=pd.DataFrame({'ass_prob':ass_prob})['ass_prob'][idxmax]


            overall_slide_label_probs=(pd.DataFrame({'classlabel':classlabel})['classlabel'].value_counts()/pd.DataFrame({'classlabel':classlabel})['classlabel'].count()).to_list()

            ass_overall_slide_label_prob=(pd.DataFrame({'classlabel':classlabel})['classlabel'].value_counts()/pd.DataFrame({'classlabel':classlabel})['classlabel'].count())[cls_slide_lbl]

            above_thresh=False
            if ass_overall_slide_label_prob > self.thresh:
                above_thresh=True

           
            slide_dataframe=pd.DataFrame(data={
                               'case_id': case_id,
                               'slide_id': unique_item,
                               'predicted_class_slide_label':cls_slide_lbl,
                               'true_class':true_cls, 
                               'associated_overall_slide_label_probability': ass_overall_slide_label_prob,
                               'above_threshold':above_thresh,
                               'class_slide_label_counts':[cls_slide_lbl_counts],                  
                               'overall_slide_label_probabilities' :[overall_slide_label_probs],
                               'minimum_slide_label_prob':min_slide_prob,
                               'maximum_slide_label_prob':max_slide_prob,                           

                              })   
            
            df_list.append(slide_dataframe)

        slide_dataframe = pd.concat(df_list)



        return tile_dataframe, slide_dataframe


    def get_prediction_per_case(self, ids=None , tile_level=True):

        case_ids=ids
        slide_ids=None
        if case_ids:                
            if type(case_ids) is not list:
                try: 
                    case_ids=[case_ids]
                except:     
                    raise NotImplementedError


            slide_ids=[]
            if isinstance(self.ds_items, pd.DataFrame):
                for iid,item in zip(self.ds_items['case_id'].to_list(),self.ds_items['fname'].to_list()):
                    
                    if iid in slide_ids:
                        ids.append(item)
                        
            if isinstance(self.ds_items, list):
                for objects in self.ds_items:
                    
                    if objects.case_id in case_ids:
                        ids.append(objects)     
            
           

            if case_ids==[] and slide_ids==[]:
                raise ValueError('None of your list of case ids is contained within the provided dataset; please check your data!')




        tile_dataframe,slide_dataframe=self.get_prediction_per_slide(
                                                       ids=slide_ids 
                                                       )      



        if tile_level:
            dataframe=tile_dataframe
            probability= 'probabilities'
            pred_label='predicted_class'
        else:
            dataframe=slide_dataframe
            probability= 'associated_overall_slide_label_probability'
            pred_label= 'predicted_class_slide_label'
        df_list=[]
        for unique_item in slide_dataframe['case_id'].unique():
            classlabel=[]
            ass_prob=[]
            ids=[]

            for label, prob, thr, iid  in zip(dataframe.loc[dataframe['case_id'] == unique_item, pred_label],
                        dataframe.loc[dataframe['case_id'] == unique_item, probability],
                         dataframe.loc[dataframe['case_id'] == unique_item, 'above_threshold'] , 
                         dataframe.loc[dataframe['case_id'] == unique_item, 'true_class']           ):  
                if tile_level:
                    prob=prob[np.argmax(prob)]

                if self.exclude_failed:
                    if thr > self.thresh:
                        classlabel.append(label)
                        ass_prob.append(prob)
                else:
                    classlabel.append(label)
                    ass_prob.append(prob)

                ids.append(iid)    
            cls_case_lbl_counts=pd.DataFrame({'classlabel':classlabel})['classlabel'].value_counts().to_list()
            cls_case_lbl=pd.DataFrame({'classlabel':classlabel})['classlabel'].value_counts().idxmax()    

            idxmin=pd.DataFrame({'ass_prob':ass_prob})['ass_prob'].idxmin()
            idxmax=pd.DataFrame({'ass_prob':ass_prob})['ass_prob'].idxmax()
            min_case_prob=pd.DataFrame({'ass_prob':ass_prob})['ass_prob'][idxmin]
            max_case_prob=pd.DataFrame({'ass_prob':ass_prob})['ass_prob'][idxmax]

            overall_case_label_probs=(pd.DataFrame({'classlabel':classlabel})['classlabel'].value_counts()/pd.DataFrame({'classlabel':classlabel})['classlabel'].count()).to_list()

            ass_overall_case_label_prob=(pd.DataFrame({'classlabel':classlabel})['classlabel'].value_counts()/
                                         pd.DataFrame({'classlabel':classlabel})['classlabel'].count())[cls_case_lbl]


            above_thresh=False
            if ass_overall_case_label_prob > self.thresh:
                above_thresh=True
            true_cls=ids[0]
            case_dataframe=pd.DataFrame(data={
                                   'case_id': unique_item,
                                   'predicted_class_case_label':cls_case_lbl,
                                   'true_class':true_cls,
                                   'associated_overall_case_label_probability': ass_overall_case_label_prob,
                                   'above_threshold':above_thresh,
                                   'class_case_label_counts':[cls_case_lbl_counts],                  
                                   'overall_case_label_probabilities' :[overall_case_label_probs],
                                   'minimum_case_label_prob':min_case_prob,
                                   'maximum_case_label_prob':max_case_prob,        

                                  })   

            df_list.append(case_dataframe)

        case_dataframe = pd.concat(df_list)      
    
        return tile_dataframe, slide_dataframe, case_dataframe