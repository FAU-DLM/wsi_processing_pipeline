class NameGetter(object):
    '''This class provides utility functions around the issue of naming files and directories from objects 
    found in any path'''
    
    
    def __init__(self,
                 path=None, 
                 path_parts=None, 
                 n_split_returns=None, 
                 splitter=None, 
                 joiner=None, 
                 rm_correctors=None, 
                 path_to_patient_mapping=None,
                 patient_to_path_mapping=None,
                 names=None,
                 labels=None, 
                 keys=None):      
           
               
         
        self.path=path 
        self.path_parts=path_parts
        self.n_split_returns=n_split_returns 
        self.splitter=splitter 
        self.joiner=joiner 
        self.rm_correctors= rm_correctors                 
        self.path_to_patient_mapping= path_to_patient_mapping
        self.patient_to_path_mapping= patient_to_path_mapping
        self.names=names
        self.labels=labels 
        self.keys=keys
        
    def set_patient_id(self, path=None, patient_ids=None, key='path'):        
        if path==None:
            path=self.path
        if not isinstance(path, list):
            path=[path]
        if patient_ids==None:
            print('Patient ID has not been specified!')
            patient_ids=[None] * len(path)  
        zipbObj = zip(path, patient_ids)
        # Create a dictionary from zip object
        self.path_to_patient_mapping=dict(zipbObj)
        if key=='path':    
            return self.path_to_patient_mapping
        else:
            self.patient_to_path_mapping={l:n for n,l in self.path_to_patient_mapping.items()}
            return self.patient_to_path_mapping
    
    def get_patient_id(self, path, patient_ids=None ):    
        if self.path_to_patient_mapping:     
            if isinstance(path, list):
                return [self.path_to_patient_mapping[p] for p in path]
            else:
                return self.path_to_patient_mapping[path]
        else:
            self.set_patient_id(path, patient_ids)
            if isinstance(path, list):
                return [self.path_to_patient_mapping[p] for p in path]
            else:
                return self.path_to_patient_mapping[path]
            
       
    def create_named_object(self, path, patient_id, case_id, slide_id, classification_label):
        if not isinstance(path, list):
            path=[path]
        if not isinstance(patient_id, list):
            patient_id=[patient_id] 
        if not isinstance(case_id, list):
            case_id=[case_id]    
        if not isinstance(slide_id, list):
            slide_id=[slide_id] 
        if not isinstance(classification_label, list):
            classification_label=[classification_label]     
        
        named_objects=[NamedObject().create(path=p,
                 patient_id=p_id,
                 case_id=c_id,
                 slide_id=s_id,
                 classification_label=c_l)for p, p_id, c_id, s_id, c_l in tqdm(zip(path,patient_id,case_id,slide_id, classification_label))]
        
        return named_objects
    
    def create_named_object_from_path(self,
                                      path=None,
                                      patient_id_getter=None,
                                      case_id_getter=None,
                                      slide_id_getter=None,
                                      classification_label_getter=None):
        presetter(variables={'path':path                             
                             },
                             cls=self
                             )
        precheck(types=['path'], 
                      path={'path':self.path
                           },
                     cls=self)
        
        paths, patient_id, case_id, slide_id, classification_label =self.get_ids_from_path(path=self.path,
                                                            patient_id_getter=patient_id_getter,
                                                            case_id_getter=case_id_getter,
                                                            slide_id_getter=slide_id_getter,
                                                            classification_label_getter=classification_label_getter)
       
           
        named_objects=self.create_named_object(paths, patient_id, case_id, slide_id, classification_label)
        
        return named_objects            
    
    
    def get_ids_from_path(self, 
                          path=None,
                          patient_id_getter=None,
                          case_id_getter=None,
                          slide_id_getter=None,
                          classification_label_getter=None):
        
        presetter(variables={'path':path                             
                             },
                             cls=self
                             )
        precheck(types=['path'], 
                      path={'path':self.path
                           },
                     cls=self)        
        if not isinstance(path, list):
            path=[path]
        if not patient_id_getter:
            patient_id = self.get_patient_id(self.path)
        if patient_id_getter:
            patient_id=patient_id_getter(self.path)
        if not case_id_getter:
            case_id = [None] * len(path)     
        if case_id_getter:
            case_id=case_id_getter(self.path)
        if not slide_id_getter:            
            slide_id = [None] * len(path)  
        if slide_id_getter:            
            slide_id=slide_id_getter(self.path)
        if not classification_label_getter:            
            classification_label = [None] * len(path) 
        if classification_label_getter:            
            classification_label = classification_label_getter(self.path)              
        
        return self.path, patient_id, case_id, slide_id, classification_label
    
    
    def create_named_object_from_df(self, df):
        paths, patient_id, case_id, slide_id, classification_label = self.get_ids_from_df(df)            
        named_objects=self.create_named_object(paths, patient_id, case_id, slide_id, classification_label)        
        return named_objects   
        
    def get_ids_from_df(self,df=None):
        path = df.path.tolist()
        patient_id = df.case.patient_id.tolist()
        case_id = df.case_id.tolist()
        slide_id = df.slide_id.tolist()
        classification_label = df.classification_label.tolist()
        #if not any(isinstance(el, NoneType) for el in case_id) and not any(isinstance(el, NoneType) for el in slide_id):
        return path, patient_id, case_id, slide_id, classification_label
        #elif any(isinstance(el, NoneType) for el in case_id) and not any(isinstance(el, NoneType) for el in slide_id):    
        #    return path, patient_id, None, slide_id    
     
    
    def get_name_from_path(self, 
                           path=None, 
                           path_parts=None, 
                           n_split_returns=None, 
                           splitter=None, 
                           joiner=None, 
                           unique=False, 
                           rm_correctors=None,
                           keep_split=False,
                           regex=None,                           
                           match=False):          
        '''Function that takes         
        "path": pathlib or string object or as list
        "path_parts: integer or as alist of integers defining parts of splitted path"
        "n_split_returns: Number of splitted parts to return of the modified path defined as integer or list of integers
        "splitter" and "joiner" strings for splitting and joining the path_parts
        "unique": boolean for getting modified processed paths as set or pure list: may contain duplicate entries
        "rm_correctors" string or list of strings to strip of off the modified processed path adjusted with the parameter
        "remove:"boolean whether to just strip the rm_corrector off or to leave out the complete path_part containing the rm_corrector
        "regex": is a regular expressions pattern used with fast.ais class RegexLabeller--> this overwrites the other methods!!
                additionally ajustable with 
        "match": boolean whether re.match or re.search is used within the RegexLabeller  
        
        
        returns:
            a string object modified from initial path_parts            
    
        '''
        presetter(variables={'path':path, 
                             'path_parts':path_parts, 
                             'n_split_returns':n_split_returns, 
                             'splitter':splitter, 
                             'joiner':joiner, 
                             'rm_correctors':rm_correctors
                             },
                             cls=self
                             )
        precheck(types=['path', 'string', 'integer', 'bool'], 
                      path={'path':self.path
                           }, 
                      string={'splitter':self.splitter,
                              'joiner': self.joiner ,
                             'rm_correctors': self.rm_correctors
                             },  
                      integer={'path_parts':self.path_parts,
                               'n_split_returns':self.n_split_returns                                                              
                              },                                        
                      bools={'unique':unique, 
                             'keep_split':keep_split,
                             'match':match
                            },
                     cls=self)
        
                
                    
        if isinstance(self.joiner, list): 
            if len(self.splitter)==1:
                self.joiner=self.joiner[0]
            else:
                raise NotImplementedError
                
        if isinstance(self.splitter, list): 
            if len(self.splitter)==1:
                self.splitter=self.splitter[0]
            else:
                raise NotImplementedError   
        
        if isinstance(path, list)  or isinstance(path,fastcore.foundation.L):
           
            return_path=self.get_names_from_paths(paths=self.path, 
                                      path_parts=self.path_parts, 
                                      splitter=self.splitter, 
                                      joiner=self.joiner, 
                                      n_split_returns=self.n_split_returns,
                                      unique=unique,
                                      rm_correctors=self.rm_correctors,
                                      regex=regex,
                                      keep_split=keep_split
                                                 )            
            
        else: 
            
            if regex:                 
                f= RegexLabeller(regex, match=match)
                return_path = f(str(self.path)) 
                
            else:               
                try: 
                    
                    return_path=self.joiner.join([self.path.parts[i] for i in self.path_parts]).split(self.splitter) 
                    
                    if not isinstance(self.rm_correctors, NoneType): 
                        if isinstance(self.rm_correctors,list): 
                            #if not any(isinstance(el, NoneType) for el in self.rm_correctors):
                        
                            parts_to_check=[return_path[i] for i in self.n_split_returns]
                            return_path,n_split_returns=self.rm_corrector(parts_to_check, keep_split=keep_split) 
                            return_path=str(Path(self.joiner.join([return_path[i] for i in n_split_returns])).with_suffix('')) 
                    else:    
                        return_path=str(Path(self.joiner.join([return_path[i] for i in self.n_split_returns])).with_suffix(''))                                            

                except:
                    raise IndexError('You may want to use the "get_path_parts_and_indices" function  \n'
                              'to precheck your path and to get the parameters "path_parts" splitter", \n'
                               '"joiner" and "n_split_returns" right!')                    
                    
            
          
        return return_path
    
    def rm_corrector(self, parts_to_check, keep_split=False):
        
        '''
        helper function for get_name_from_path:
        takes the path part to be adjusted if 
        "rm_correctors"(list of strings or just string object)
        is set        
        returns list of path objects and list of integers
        '''
        if any(isinstance(el, NoneType) for el in self.rm_correctors):
            raise ValueError('You cannot use this function by itself it is used within other functions!')
        
        precheck(types=[ 'bool'],                       
                      bools={'keep_split':keep_split}
                     )          
             
        rm_correctors=L(self.rm_correctors)
        
        rms='(?:% s)' % '|'.join(rm_correctors)                       
        
        if keep_split:            
            return_path=[re.sub(rms, '', parts)   for parts in parts_to_check]             
            return_path = [x for x in return_path if x]           
        else:
            return_path=[parts for parts in parts_to_check if not re.compile(rms).search(parts) ]
            
        n_split_returns=list(range(len(return_path)))          
        
        return return_path, n_split_returns
        
    def get_names_from_paths(self, 
                             paths=None, 
                             path_parts=None, 
                             n_split_returns=None, 
                             splitter=None, 
                             joiner=None, 
                             unique=False, 
                             regex=None,
                             rm_correctors=None,
                             keep_split=False,
                             match=False):
        
        '''
        
        
        
        
        '''
        
        presetter(variables={'path':paths, 
                             'path_parts':path_parts, 
                             'n_split_returns':n_split_returns, 
                             'splitter':splitter, 
                             'joiner':joiner, 
                             'rm_correctors':rm_correctors},
                             cls=self
                             )  
        precheck(types=['path', 'string', 'integer', 'bool'],
                      path={'path':self.path},
                      string={'splitter':self.splitter,
                              'joiner': self.joiner ,
                             'rm_correctors': self.rm_correctors
                             },  
                      integer={'path_parts':self.path_parts,
                               'n_split_returns':self.n_split_returns                                                              
                              },                                        
                      bools={'unique':unique, 
                             'keep_split':keep_split,
                             'match':match
                            },
                     cls=self) 
        
        paths= [Path(p) if not isinstance(p,pathlib.PosixPath) else p for p in self.path ]            
               
        if unique:            
            result=list(set([self.get_name_from_path(path=ps,
                                                     path_parts=self.path_parts, 
                                                     splitter=self.splitter, 
                                                     joiner=self.joiner, 
                                                     n_split_returns=self.n_split_returns, 
                                                     unique=unique,
                                                     regex=regex,
                                                     rm_correctors=self.rm_correctors,
                                                     keep_split=keep_split,
                                                     match=match) for ps in paths]))
        else:
            result=[self.get_name_from_path(path=ps,
                                            path_parts=self.path_parts, 
                                            splitter=self.splitter, 
                                            joiner=self.joiner, 
                                            n_split_returns=self.n_split_returns, 
                                            unique=unique,
                                            regex=regex,
                                            rm_correctors=rm_correctors,
                                            keep_split=keep_split,
                                            match=match) for ps in paths]
            
        return result    
    
    
    def get_path_parts_and_indices(self, path=None): 
        presetter(variables={'path':path},cls=self)
        
        precheck(types=['path'], 
                      path={'path':self.path} ,
                      classes=self
                     )                              
                          
        if isinstance(self.path, list):
            path_parts=[] 
            res=[]              
            for parts in self.path:               
                parts=list(parts.parts)           
                path_parts.append(parts)
                res.append({i:e for i,e in enumerate(parts)})                             
        else:             
            path_parts=list(self.path.parts)
            res=({i:e for i,e in enumerate(path_parts)})      
        
        return res
        
    def get_path(self):
        precheck(types=['path'], 
                      path={'path':self.path},
                      classes=self
                     )         
       
        return self.path          
    
    def num2lbs(self, 
                labels=None, 
                keys=None):
        presetter(variables={'labels':labels, 
                             'keys':keys,                               
                             },
                             cls=self
                             )
        precheck(types=['str-or-int'],                       
                      str_or_int={'labels':self.labels,
                                      'keys':self.keys                                                                          
                                      },
                     cls=self) 
        
        if self.keys==[None]:
            num2lbs = { i : self.labels[i] for i in range(0, len(self.labels) ) }
            
        elif self.keys!=[]:
            zipbObj = zip( self.keys, self.labels)
        # Create a dictionary from zip object
            num2lbs = dict(zipbObj)
        return num2lbs

    def lbs2num(self):
        return{l:n for n,l in self.num2lbs().items()}    
        
    def label_func(self,
                   names=None,
                   labels=None, 
                   keys=None, 
                   return_list=False, 
                   return_name=False,                                     
                   match=False,
                   multicategory=False):
        
        presetter(variables={'labels':labels, 
                             'keys':keys,
                             'names':names,
                             },
                             cls=self
                             )
                 
        
        precheck(types=['str-or-int', 'bool'],                     
                      str_or_int={'labels':self.labels,
                                   'keys':self.keys,                                      
                                      },
                     bools={'return_list':return_list,
                            'return_name': return_name,                            
                            'match':match,
                            'multicategory':multicategory},
                    cls=self) 
        
        if isinstance(self.keys, NoneType):
            self.keys=[self.keys]
        if multicategory:
            raise NotImplementedError
            
        else:        
                
            if not any(isinstance(el, list) for el in names):
                    
                    if return_list:
                        if return_name:
                            res=[[name] if name in self.labels else [NoneType] for name in names  ]
                        else:
                            res=[[self.lbs2num()[name]] if name in self.labels else [NoneType] for name in names ]
                    else:
                        if return_name: 
                            res=[name if name in self.labels else NoneType for name in names ]
                        else:
                            res=[self.lbs2num()[name] if name in self.labels else NoneType for name in names ]     

            elif any(isinstance(el, list) for el in names):
                    raise NotImplementedError
            else:
                    raise AssertionError            
                
            self.data_labels=res
            return res