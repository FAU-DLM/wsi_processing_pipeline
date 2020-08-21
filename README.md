# wsi_processing_pipeline
This library helps with the key pre- and postprocessing steps necessary to use whole-slide images in deep-learning/ai projects.

Current update (2020.08.21):
The new preprocessing structure is now implemented and working.
The hierarchy is the following:
PatientManager -> Patient -> Case -> WholeSlideImage -> RegionOfInterest -> Tile

NamedObject, ObjectManager and WsiOrRoiObject are now deprecated and will be removed soon.

For a current example pipeline checkout the following notebook:
https://github.com/ChristophNeuner/pituitary_gland_adenomas/blob/master/network-entity_classification_fastai2.ipynb
