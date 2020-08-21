#####
#
#
#
#
# These functions are deprecated and will be removed in the near future.
# 
#
#
#
#
#
#
#####






import pathlib
from pathlib import Path
import fastcore

def presetter(variables={}, cls=None):
        ''' This is just to preset variables in the functions within the class; no checks are made here'''
        if cls is None:raise ValueError
        if variables!={} and isinstance(variables,dict):    
            for key,value in variables.items():
                if value:
                    setattr(cls, key, value)
        else:raise ValueError    
            
def precheck(**kwargs): 
        '''This function prechecks the different types within the class:
        NoneType is not checked for!!
        "types": divided into "path" for path-like objects, "str-int" for string or integer like objects and "bool" for boolian types.
        
        within these 
        
        "path" is a dictionary named "path" and has the keys
                path= takes string like or PosixPath objects or which may reside within a list.
        
        "string": is a dictionary containing attributes to be checked whether they are of type string or as alist of strings
                 they can than be added to self or not.
        
        "integer":is a dictionary containing attributes to be checked whether they are of type int or as a list of ints
                 they can than be added to self or not.
        
        str-or-int: is a dictionary containing attributes to be checked whether they are of type int or string or as a list of ints
                     or strings they can than be added to self or not.        
            
        "cls": if this is added path, string, integer or str-or-int attributes will be set and added to this!.. for instance if you
               add the self attribute"
               
        "bool" parameters should be set as type boolian 
        
            '''
        results=[]
        if 'path' in kwargs['types']: 
            for key, value in kwargs['path'].items():
                
                #    raise ValueError('Please specify a path or a list of paths as "string" or "pathlib.PosixPath" objects')
                #print(type(value))
                if isinstance(value,str):
                    value=Path(value)
                elif isinstance(value, pathlib.PosixPath):
                    pass                
                elif isinstance(value, list):
                    if any(isinstance(el, str) for el in value):
                        value=[Path(el) for el in value] 
                    elif any(isinstance(el, pathlib.PosixPath) for el in value):
                        value=[el for el in value]
                    else:
                        raise ValueError('Please specify a path or a list of paths as "string" or "pathlib.PosixPath" objects')      
                     
                else:
                    raise ValueError('Please specify a path or a list of paths as "string" or "pathlib.PosixPath" objects')    
                           
                #setattr(kwargs['classes'], key, value)
                if 'cls' in kwargs['types']:
                    setattr(kwargs['cls'], key, value)
                else:
                    results.append(value)
                
        if 'string' in kwargs['types']:             
            for key, value in kwargs['string'].items(): 
                if value is None:
                    if 'cls' in kwargs['types']:
                        setattr(kwargs['cls'], key, [value])
                    else:    
                        results.append([value] )
                    
                elif not isinstance(value, list) and not isinstance(value,fastcore.foundation.L):     
                    if isinstance(value, str) :
                            pass                           
                        #
                    else:    
                        raise ValueError(f'Please specify the parameter {key} as a string or as a list of strings and not as {type(value)})!') 
                    
                    if 'cls' in kwargs['types']:
                        setattr(kwargs['cls'], key, [value])
                    else:    
                        results.append([value] )      

                elif isinstance(value, list) or isinstance(value,fastcore.foundation.L):                   
                          
                    ls=[]
                    if any(el is None for el in value):
                        for parts in value:
                            ls.append(parts)
                    
                    else:
                        for parts in value:                
                            #if key=='labels' or key =='keys':

                            if isinstance(parts, str):                                                                   
                                        #pass
                                ls.append(parts)                          
                            else:
                                raise ValueError(f'Please specify the parameter {key} as a string/integer or as a list of strings/integers!')
                    #ls=[item for item in ls]  
                    if 'cls' in kwargs['types']:
                        setattr(kwargs['cls'], key, ls)
                    else:    
                        results.append(ls)                                                                                                  
                    #setattr(kwargs['classes'], key, ls)              
                
        if 'integer' in kwargs['types']:             
            for key, value in kwargs['integer'].items():
                if value is None:
                    if 'cls' in kwargs['types']:
                        setattr(kwargs['cls'], key, [value])
                    else:    
                        results.append([value] )
                elif not isinstance(value, list) and not isinstance(value,fastcore.foundation.L): 
                    try:
                            value = int(value)
                    except:                        
                            raise ValueError(f'Please specify the parameter {key} as an integer or as a list of integers!')        
                                                                  
                       # setattr(kwargs['classes'], key, [value])  
                    if 'cls' in kwargs['types']:
                        setattr(kwargs['cls'], key,[value])
                    else:    
                        results.append([value] )          

                elif isinstance(value, list) or isinstance(value,fastcore.foundation.L):                   
                          
                    ls=[]
                    if any(el is None for el in value):
                        for parts in value:
                            ls.append(parts)
                    
                    else:
                        for parts in value: 
                            try:
                                    parts= int(parts)
                                    ls.append(parts)
                            except:                        
                                    raise ValueError(f'Please specify the parameter {key} as an integer or as a list of integers!')   
                         
                    #ls=[item for item in ls]  
                    if 'cls' in kwargs['types']:
                        setattr(kwargs['cls'], key, ls)
                    else:
                        results.append(ls)                  
                
                        
        if 'str-or-int' in kwargs['types']:             
            for key, value in kwargs['str_or_int'].items():
                if value is None:
                    if 'cls' in kwargs['types']:
                        setattr(kwargs['cls'], key, [value])
                    else:    
                        results.append([value] )
                        
                elif not isinstance(value, list) and not isinstance(value,fastcore.foundation.L):                                   
                
                    if isinstance(value, str) or isinstance(value, int) :                        
                            #pass
                        if 'cls' in kwargs['types']:
                            setattr(kwargs['cls'], key, [value])    
                        else:
                            results.append([value])                   
                    else:
                        raise ValueError(f'Please specify the parameter {key} as a string/integer or as a list of strings/integers!')                           
                elif isinstance(value, list) or isinstance(value,fastcore.foundation.L):
                    ls=[]
                    if any(el is None for el in value):
                        for parts in value:
                            ls.append(parts)
                    
                    else:
                        for parts in value:                
                        #if key=='labels' or key =='keys':
                            if isinstance(parts, str) or isinstance(parts, int):                                                                   
                                        #pass
                                ls.append(parts)                          
                            else:
                                raise ValueError(f'Please specify the parameter {key} as a string/integer or as a list of strings/integers!')
                    #ls=[item for item in ls]        
                    if 'cls' in kwargs['types']:
                        setattr(kwargs['cls'], key, ls)
                    else:
                        results.append(ls)
                    
        if 'bool' in kwargs['types']:            
            for key, value in kwargs['bools'].items():                 
                if not isinstance(value, bool):
                    raise ValueError(f'Please specify the parameter {key} as type "bool"; "True" or "False"') 
        
        return results    