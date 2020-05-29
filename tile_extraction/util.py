# ------------------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------
import typing
from typing import Callable, Union, Dict, List, Tuple
import pathlib
from pathlib import Path
import datetime
import numpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import fastai

import sys
sys.path.append("../")
import tile_extraction
from tile_extraction import tiles
from tiles import *

# If True, display additional NumPy array stats (min, max, mean, is_binary).
ADDITIONAL_NP_STATS = False


def show_wsi_with_rois(wsi_path:pathlib.Path, 
                       rois:List[RegionOfInterest], 
                       figsize:Tuple[int] = (10,10), 
                       scale_factor:int = 32):
        """    
        Loads a whole slide image, scales it down, converts it into a numpy array 
        and displays it with a grid overlay for all tiles,
        that passed scoring to visualize which tiles e.g. "tiles.WsiOrROIToTilesMultithreaded" calculated as worthy to keep.
        Arguments:
            figsize: Size of the plotted matplotlib figure containing the image.
            scale_factor: The larger, the faster this method works, but the plotted image has less resolution.
            tilesummary: a TileSummary object of one wsi
            wsi_path: Path to a whole-slide image
            df_tiles: A pandas dataframe from e.g. "tiles.WsiOrROIToTilesMultithreaded" 
                        with spacial information about all tiles           
        """
        wsi_pil, large_w, large_h, new_w, new_h, best_level_for_downsample = tiles.wsi_to_scaled_pil_image(wsi_path,
                                                                                            scale_factor=scale_factor,
                                                                                            level=0)
        wsi_np = pil_to_np_rgb(wsi_pil)
        boxes =[]
        for roi in rois:
            roi_adj = roi.change_level_deep_copy(new_level=best_level_for_downsample)
            box = np.array([roi_adj.x_upper_left, roi_adj.y_upper_left, roi_adj.width, roi_adj.height])
            boxes.append(box)
        show_np_with_bboxes(wsi_np, boxes, figsize)
        
        
def adjust_level(value_to_adjust:int, from_level:int, to_level:int)->int:
    """
    Arguments: 
        value_to_adjust: pixel size that shall be adjusted
        from_level: The whole-slide image level the <value_to_adjust> is currently in
        to_level: The level to which the <value_to_adjust> shall be transformed to
    Returns:
        adjusted pixel size (rounded to int)
    """
    if(from_level < to_level):
        return round(value_to_adjust/(2**(to_level-from_level)))
    if(from_level > to_level):
        return round(value_to_adjust*(2**(from_level-to_level)))
    else:
        return value_to_adjust


def safe_dict_access(dict:Dict, key):
    """
    returns None if anything goes wrong at getting the value from the dict, else returns the value
    """
    try:
        return dict[key]
    except:
        return None

def show_np_with_bboxes(img:numpy.ndarray, bboxes:List[numpy.ndarray], figsize:tuple=(10,10)):
    """
    Arguments:
        img: img as numpy array
        bboxes: List of bounding boxes where each bbox is a numpy array: 
                array([ x-upper-left, y-upper-left,  width,  height]) 
                e.g. array([ 50., 211.,  17.,  19.])
    """    
    # Create figure and axes
    fig,ax = plt.subplots(1,1,figsize=figsize)    
    # Display the image
    ax.imshow(img)    
    # Create a Rectangle patch for each bbox
    for b in bboxes:
        rect = matplotlib.patches.Rectangle((b[0],b[1]),b[2],b[3],linewidth=1,edgecolor='r',facecolor='none')    
        # Add the patch to the Axes
        ax.add_patch(rect)    
    plt.show() 


def show_np(np):
    return util.np_to_pil(np)

def show_multiple_images(paths:list, rows = 3, figsize=(128, 64)):
    """
    Args:
        paths: A list of paths to images.
    """
    imgs = [fastai.vision.open_image(p) for p in paths]
    fastai.vision.show_all(imgs=imgs, r=rows, figsize=figsize)
    
def show_multiple_images_big(paths:list, axis_off:bool = False):
    """
    Args:
        paths: A list of paths to images.
    """
    for p in paths:
        plt.imshow(mpimg.imread(str(p)))
        if(axis_off):
            plt.axis('off')
        plt.show()


def pil_to_np_rgb(pil_img):
  """
  Convert a PIL Image to a NumPy array.

  Note that RGB PIL (w, h) -> NumPy (h, w, 3).

  Args:
    pil_img: The PIL Image.

  Returns:
    The PIL image converted to a NumPy array.
  """
  #t = Time()
  rgb = np.asarray(pil_img)
  #np_info(rgb, "RGB", t.elapsed())
  return rgb


def np_to_pil(np_img):
  """
  Convert a NumPy array to a PIL Image.

  Args:
    np_img: The image represented as a NumPy array.

  Returns:
     The NumPy array converted to a PIL Image.
  """
  if np_img.dtype == "bool":
    np_img = np_img.astype("uint8") * 255
  elif np_img.dtype == "float64":
    np_img = (np_img * 255).astype("uint8")
  return Image.fromarray(np_img)


def np_info(np_arr, name=None, elapsed=None):
  """
  Display information (shape, type, max, min, etc) about a NumPy array.

  Args:
    np_arr: The NumPy array.
    name: The (optional) name of the array.
    elapsed: The (optional) time elapsed to perform a filtering operation.
  """

  if name is None:
    name = "NumPy Array"
  if elapsed is None:
    elapsed = "---"

  if ADDITIONAL_NP_STATS is False:
    print("%-20s | Time: %-14s  Type: %-7s Shape: %s" % (name, str(elapsed), np_arr.dtype, np_arr.shape))
  else:
    # np_arr = np.asarray(np_arr)
    max = np_arr.max()
    min = np_arr.min()
    mean = np_arr.mean()
    is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
    print("%-20s | Time: %-14s Min: %6.2f  Max: %6.2f  Mean: %6.2f  Binary: %s  Type: %-7s Shape: %s" % (
      name, str(elapsed), min, max, mean, is_binary, np_arr.dtype, np_arr.shape))


def display_img(np_img, text=None, font_path="/Library/Fonts/Arial Bold.ttf", size=48, color=(255, 0, 0),
                background=(255, 255, 255), border=(0, 0, 0), bg=False):
  """
  Convert a NumPy array to a PIL image, add text to the image, and display the image.

  Args:
    np_img: Image as a NumPy array.
    text: The text to add to the image.
    font_path: The path to the font to use.
    size: The font size
    color: The font color
    background: The background color
    border: The border color
    bg: If True, add rectangle background behind text
  """
  result = np_to_pil(np_img)
  # if gray, convert to RGB for display
  if result.mode == 'L':
    result = result.convert('RGB')
  draw = ImageDraw.Draw(result)
  if text is not None:
    font = ImageFont.truetype(font_path, size)
    if bg:
      (x, y) = draw.textsize(text, font)
      draw.rectangle([(0, 0), (x + 5, y + 4)], fill=background, outline=border)
    draw.text((2, 0), text, color, font=font)
  result.show()


def mask_rgb(rgb, mask):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  #t = Time()
  result = rgb * np.dstack([mask, mask, mask])
  #np_info(result, "Mask RGB", t.elapsed())
  return result


class Time:
  """
  Class for displaying elapsed time.
  """

  def __init__(self):
    self.start = datetime.datetime.now()

  def elapsed_display(self):
    time_elapsed = self.elapsed()
    print("Time elapsed: " + str(time_elapsed))

  def elapsed(self):
    self.end = datetime.datetime.now()
    time_elapsed = self.end - self.start
    return time_elapsed
