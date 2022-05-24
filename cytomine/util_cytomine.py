import sys
sys.path.append("../")
from shared import roi
from tile_extraction import util

from typing import Dict, List, Callable, Tuple, Union, Sequence

import cytomine
from cytomine import Cytomine
from cytomine.models import CurrentUser, Project, ProjectCollection, ImageInstanceCollection, AnnotationCollection
from cytomine.models import Annotation, AnnotationTerm
from cytomine.models.ontology import Ontology, Term, RelationTerm, TermCollection, OntologyCollection

import pathlib
from pathlib import Path
import openslide
import shapely
from shapely import wkt


def lists_overlap(a:List, b:List)->bool:
    """
    Returns True, if the two lists have an overlap, else False
    """
    return bool((set(a), set(b)))

def get_annotations(image:cytomine.models.ImageInstance)->cytomine.models.AnnotationCollection:
    """
    Returns all annotations of the image.
    """
    annotations = AnnotationCollection()
    annotations.image = image.id
    return annotations.fetch()

def get_annotations_with_term_filter(image:cytomine.models.ImageInstance, 
                    included_terms:List[str], 
                    excluded_terms:List[str])->List[cytomine.models.Annotation]:
    """
    Arguments:
        image: cytomine ImageInstance
        included_terms: Only annotations, which have all specified term names, will be returned
        excluded_terms: Only annotations, which do not have any of these terms, will be returned
    Returns:
         List of annotations
    
    """
    annotations_filtered = []
    for a in get_annotations(image=image):
        terms = util_cytomine.get_terms_of_annotation(a.fetch())
        term_names = [t.name for t in terms]
        if(not lists_overlap(a=excluded_terms, b=term_names) and set(included_terms).issubset(term_names)):
            annotations_filtered.append(a)
    return annotations_filtered



def get_cytomine_image_instance_for_wsi_name(wsi_name:str, 
                                             project:cytomine.models.project.Project=None)\
                                            ->List[cytomine.models.image.ImageInstance]:
    """
    Iterates through the given project and looks for the given wsi.
    If no project is specified, all projects the user has access to are iterated.
    All matches are returned.
    Arguments:
        wsi_name as a string with file extension: e.g. casus_1.ndpi
        project: cytomine.models.project.Project instance
    Returns:
        List of all cytomine.models.image.ImageInstance objects that match. May be an empty list.
    """
    projects = None
    if(project is not None):
        projects = [project]
    else:
        projects = ProjectCollection().fetch()
    
    found_images = []
    for p in projects:
        for i in ImageInstanceCollection().fetch_with_filter("project", p.id):
            if(i.filename == wsi_name):
                found_images.append(i)
    return found_images
    
def get_project_for_image(image:cytomine.models.image.ImageInstance, 
                          projects:Union[List[cytomine.models.project.Project], 
                                         cytomine.models.project.ProjectCollection]=None)\
                                        ->Union[cytomine.models.project.Project, None]:
    """
    Iterates through the given projects and looks for the image's project.
    If no project is specified, all projects the user has access to are iterated.
    Arguments:
        image: cytomine.models.image.ImageInstance
        projects: If None, 
    Returns:
        Matching cytomine.models.project.Project object, else None
    """
    
    if(projects is None):
        projects = ProjectCollection().fetch() 
    
    for p in projects:
        if(p.id == image.project):
            return p
    return None
    
    
def delete_all_annotations(image:cytomine.models.ImageInstance):
    for a in get_annotations(image=image):
        a.delete()
        
def delete_annotations(annotations:Union[List[cytomine.models.Annotation], cytomine.models.AnnotationCollection]):
    for a in annotations:
        a.delete()

def get_ontology_by_id(ontology_id:int)->Union[cytomine.models.Ontology, None]:
    """
    Looks for the given ontology in all ontologies.
    Returns None if it does not exist.
    """
    for o in OntologyCollection().fetch():
        if(o.id == ontology_id):
            return o
    return None        
        
def get_term_by_name(term_name:str, ontology_id:int)->cytomine.models.Term:
    """
    Looks for the term in all existing ontologies.
    If it does exist in the given ontology_id => returns existing term
    If it does exist in another ontology => creates a new term with the same color. (cytomine does not allow to add a term of
        a foreign ontology to an annotation and does also not allow to add a term from one ontology to another with the same id)
    If it does not exist, it will be created for the ontology with the given ontology_id.
    """
    
    #first check if the given ontology exists and search in its terms
    o = get_ontology_by_id(ontology_id=ontology_id)
    if(o is None):
        raise ValueError('The given ontology_id does not exist.')
    for t in TermCollection().fetch_with_filter("ontology", o.id):
        if(t.name == term_name):
            return t
        
    #look for the term in all ontologies
    #unfortunately TermCollection().fetch() has a bug and returns 403 Forbidden even as admin
    os = OntologyCollection().fetch()
    for o in os:
        for t in TermCollection().fetch_with_filter("ontology", o.id):
            if(t.name == term_name):
                return Term(name=term_name, id_ontology=ontology_id, color=term.color).save()
    
    # could not find the term in any ontology => create it for the given ontology_id with a random color
    else:
        return Term(name=term_name, id_ontology=ontology_id, color=get_random_hexadecimal_rgb_color()).save()
    
def get_random_hexadecimal_rgb_color()->str:
    import random
    r = lambda: random.randint(0,255)
    return('#%02X%02X%02X' % (r(),r(),r()))

def add_terms_to_annotation(annotation:cytomine.models.Annotation, terms:List[str], ontology_id:int):
    """
    Arguments:
    annotation: cytomine annotation object
    terms: list of term names
    """
    for t in terms:
        term = get_term_by_name(term_name=t, ontology_id=ontology_id)
        AnnotationTerm(annotation.id, term.id).save()
        
def get_wsi_path_from_cytomine_image_instance(i:cytomine.models.image.ImageInstance, wsi_paths:List[pathlib.Path]):
    wsi_path = None
    for wp in wsi_paths:
        if(i.filename == wp.name):
            return wp
                
def get_terms_of_annotation(annotation:cytomine.models.Annotation)->List[cytomine.models.AnnotationTerm]:
    # Bug in api:
    # term = AnnotationTerm().fetch(id_annotation=annotation.id, id_term=annotation.term[0])
    # using this way to get the term results in empty term.name attributes
    project = Project().fetch(id=annotation.project)
    annotation_terms = []
    for t in TermCollection().fetch_with_filter("ontology", project.ontology):
        if(t.id in annotation.term):
            annotation_terms.append(t)
    return annotation_terms

def get_image_instance_annotations_as_rois(image:cytomine.models.ImageInstance, 
                                           wsi_path:pathlib.Path)->List[roi.RegionOfInterestPolygon]:
    """
    Arguments:
        image: cytomine ImageInstance object
        wsi_path: local path to the wsi. If it not specified, the wsi will be downloaded from the server and deleted 
                    afterwards.
    Returns:
        List of wsi_processing_pipeline.shared.roi.RegionOfInterestPolygon objects
    
    """
    tmp_wsi_path = None
    if(wsi_path is None or not wsi_path.exists()):
        tmp_dir = Path('./tmp')
        tmp_dir.mkdir(exist_ok=True)
        tmp_wsi_path = tmp_dir/image.filename
        downloaded = False
        while(not downloaded):
            downloaded = image.download(dest_pattern=str(tmp_wsi_path), override=False)
    
    
    #open image
    wsi = openslide.open_slide(str(wsi_path))
    
    ##
    # fetch image annotations from the server
    ##
    annotations = AnnotationCollection()
    annotations.image = image.id
    annotations.fetch()
    
    ##
    # convert them into rois and save them in the cytomine ImageInstance object
    ##
    rois = []
    for a in annotations:
        a.fetch()                
        poly = util.switch_origin_of_shapely_polygon(polygon=wkt.loads(a.location), 
                                                polygon_level=0, 
                                                wsi_height=wsi.dimensions[1], 
                                                wsi_height_level=0)
        labels = [term.name for term in get_terms_of_annotation(annotation=a)]
        r = roi.RegionOfInterestPolygon(roi_id=a.id, 
                                vertices=util.polygon_to_numpy(poly), 
                                level=0,
                                labels=labels)
        rois.append(r)
            
    if(tmp_wsi_path is not None):
        tmp_wsi_path.delete()
    return rois