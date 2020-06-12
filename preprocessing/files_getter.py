import pathlib
import fastai2
from fastai2.vision.all import *
from tqdm import tqdm
import scandir
from typing import Dict
from wsi_processing_pipeline.preprocessing.checks import presetter, precheck

class FilesGetter(object):
        
    def get_dirs_and_files(self,
                           path=None, 
                           folder_name=None, 
                           suffix=None,
                           get_dirs=False,
                           get_files=True, 
                           recursive=True, 
                           followlinks=True, 
                           match=True,
                           file_regex=None):
        "Get all the files or folders in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified."
        
        results=precheck(types=['path', 'string', 'bool'],
                      path={'path':path} , 
                      string={'folder_name':folder_name                                                                           
                                      },
                      bools={'recursive': recursive,
                            'followlinks': followlinks,
                            'get_files': get_files,
                            'match':match}
                      )
        
        if folder_name:            
            path,folder_name = results[0],results[1]   
        else:
            path = results[0]
        folder_name=L(folder_name)    
        suffix = setify(suffix)
        suffix = {e.lower() for e in suffix}
        res = L([]) 
        if get_files:
            fls = []  
        if recursive:
            if not path.is_dir():
                return [path]                       
            
            for p,d,f in tqdm(scandir.walk(str(path), followlinks=followlinks)): # returns (dirpath, dirnames, filenames)                   
                
                #if len(folder_name) !=0: d[:] = [o for o in d if o in folder_name]
                #else:                    d[:] = [o for o in d if not o.startswith('.')]
                if len(folder_name) !=0:                   
                    
                    for fn in folder_name:                          
                        if match:  
                            pattern=fr'{os.path.sep}{fn}$'
                        else:
                            pattern=fr'{fn}'                            
                        if re.compile(pattern).search(p) and not re.compile(fr'{os.path.sep}(\.)').search(p):                               
                            res.append(Path(str(p)))
                            if get_files:
                                fls += fastai2.data.transforms._get_files(p, f, suffix)                  
                    
                else:
                    if not p.startswith('.') and not re.compile(fr'{os.path.sep}(\.)').search(p):
                        res.append(Path(p))
                        if get_files:
                            fls += fastai2.data.transforms._get_files(p, f, suffix)               
      
        else:                    
            res = [Path(o.path) for o in os.scandir(path) if o.is_dir() and not o.name.startswith('.')]            
            if len(res)==0 and path.is_dir():
                res=path
            
            if get_files:
                fs = [o.name for o in os.scandir(path) if o.is_file()]
                fls += fastai2.data.transforms._get_files(path, fs, suffix)
        
                    
        if get_files and not get_dirs:
            fls=L(fls) 
            if file_regex:                
                fls=L([Path(p) for p in fls if re.compile(file_regex).search(str(p))])        
            return fls
        elif get_files and get_dirs:
            fls=L(fls) 
            if file_regex:                
                fls=L([Path(p) for p in fls if re.compile(file_regex).search(str(p))])        
            return fls, res            
        else:            
            return res