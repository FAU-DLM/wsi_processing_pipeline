from __future__ import annotations #https://stackoverflow.com/questions/33837918/type-hints-solve-circular-dependency


from shared.case import Case


class WholeSlideImage:
    case:Case = None
    regions_of_interest:List[RegionOfInterest] = None
    slide_id:str = None
    path:pathlib.Path = None
      
    def __init__(self, slide_id:str, case:Case, path=None):
        self.region_of_interests = []
        self.case = case
        self.slide_id = slide_id
        self.path = path
        self.regions_of_interest = []
        