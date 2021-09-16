// workaround for a bug in current version: https://forum.image.sc/t/export-import-re-export-annotation-question/44075/2
def defaultColor = getColorRGB(200, 0, 0)
getAnnotationObjects().each {
   if (it.getColorRGB() == null) {
     def newColor = it.getPathClass() == null ? defaultColor : it.getPathClass().getColor()
     it.setColorRGB(newColor)
  }
}
fireHierarchyUpdate()




//filter out Tiles in annotations
def annotations = getAnnotationObjects().findAll {it.getName() == null || !it.getName().startsWith("Tile ")}
boolean prettyPrint = true
def gson = GsonTools.getInstance(prettyPrint)


import org.apache.commons.io.FilenameUtils
//get the wsi name without the type extension
def imageData = QPEx.getCurrentImageData()
def server = imageData.getServer()
String wsi_path = server.getPath()
String wsi_name = FilenameUtils.getBaseName(new File(wsi_path).getName());



def output_path = "/home/user_name/json_files/" + wsi_name + ".json"
def file = new File(output_path)

file.write(gson.toJson(annotations))

print "Done!"
