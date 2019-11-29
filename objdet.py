import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from picamera.array import PiRGBArray
import picamera
from collections import defaultdict
from io import StringIO
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import collections
import functools
# Set headless-friendly backend.
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six

import threading

from object_detection.core import standard_fields as fields

import urllib.request


_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]




width=640
height=480

quadcache="OOOO"

def urlreq(urli):
      try:
            urllib.request.urlopen(urli)
      except:
            print("error")
def visualize_boxes_and_labels_on_image_array(
            image,
            boxes,
            classes,
            scores,
            category_index,
            instance_masks=None,
            instance_boundaries=None,
            keypoints=None,
            use_normalized_coordinates=False,
            max_boxes_to_draw=20,
            min_score_thresh=.5,
            agnostic_mode=False,
            line_thickness=4,
            groundtruth_box_visualization_color='black',
            skip_scores=False,
            skip_labels=False):
      """Overlay labeled boxes on an image with formatted scores and label names.

      This function groups boxes that correspond to the same location
      and creates a display string for each detection and overlays these
      on the image. Note that this function modifies the image in place, and returns
      that same image.

      Args:
            image: uint8 numpy array with shape (img_height, img_width, 3)
            boxes: a numpy array of shape [N, 4]
            classes: a numpy array of shape [N]. Note that class indices are 1-based,
                  and match the keys in the label map.
            scores: a numpy array of shape [N] or None.      If scores=None, then
                  this function assumes that the boxes to be plotted are groundtruth
                  boxes and plot all boxes as black with no classes or scores.
            category_index: a dict containing category dictionaries (each holding
                  category index `id` and category name `name`) keyed by category indices.
            instance_masks: a numpy array of shape [N, image_height, image_width] with
                  values ranging between 0 and 1, can be None.
            instance_boundaries: a numpy array of shape [N, image_height, image_width]
                  with values ranging between 0 and 1, can be None.
            keypoints: a numpy array of shape [N, num_keypoints, 2], can
                  be None
            use_normalized_coordinates: whether boxes is to be interpreted as
                  normalized coordinates or not.
            max_boxes_to_draw: maximum number of boxes to visualize.      If None, draw
                  all boxes.
            min_score_thresh: minimum score threshold for a box to be visualized
            agnostic_mode: boolean (default: False) controlling whether to evaluate in
                  class-agnostic mode or not.      This mode will display scores but ignore
                  classes.
            line_thickness: integer (default: 4) controlling line width of the boxes.
            groundtruth_box_visualization_color: box color for visualizing groundtruth
                  boxes
            skip_scores: whether to skip score when drawing a single detection
            skip_labels: whether to skip label when drawing a single detection

      Returns:
            uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
      """
      # Create a display string (and color) for every box location, group any boxes
      # that correspond to the same location.
      box_to_display_str_map = collections.defaultdict(list)
      box_to_color_map = collections.defaultdict(str)
      box_to_instance_masks_map = {}
      box_to_instance_boundaries_map = {}
      box_to_keypoints_map = collections.defaultdict(list)
      if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
      for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                  box = tuple(boxes[i].tolist())
                  if instance_masks is not None:
                        box_to_instance_masks_map[box] = instance_masks[i]
                  if instance_boundaries is not None:
                        box_to_instance_boundaries_map[box] = instance_boundaries[i]
                  if keypoints is not None:
                        box_to_keypoints_map[box].extend(keypoints[i])
                  if scores is None:
                        box_to_color_map[box] = groundtruth_box_visualization_color
                  else:
                        display_str = ''
                        if not skip_labels:
                              if not agnostic_mode:
                                    if classes[i] in category_index.keys():
                                          class_name = category_index[classes[i]]['name']
                                    else:
                                          class_name = 'N/A'
                                    display_str = str(class_name)
                        if not skip_scores:
                              if not display_str:
                                    display_str = '{}%'.format(int(100*scores[i]))
                              else:
                                    display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
                        box_to_display_str_map[box].append(display_str)
                        if agnostic_mode:
                              box_to_color_map[box] = 'DarkOrange'
                        else:
                              box_to_color_map[box] = STANDARD_COLORS[
                                          classes[i] % len(STANDARD_COLORS)]
      personboxes=[]
      # Draw all boxes onto image.
      for box, color in box_to_color_map.items():
            ymin, xmin, ymax, xmax = box
            print("box=",box)
            personboxes.append(box)
            if instance_masks is not None:
                  vis_util.draw_mask_on_image_array(
                              image,
                              box_to_instance_masks_map[box],
                              color=color
                  )
            if instance_boundaries is not None:
                  vis_util.draw_mask_on_image_array(
                              image,
                              box_to_instance_boundaries_map[box],
                              color='red',
                              alpha=1.0
                  )
            vis_util.draw_bounding_box_on_image_array(
                        image,
                        ymin,
                        xmin,
                        ymax,
                        xmax,
                        color=color,
                        thickness=line_thickness,
                        display_str_list=box_to_display_str_map[box],
                        use_normalized_coordinates=use_normalized_coordinates)
            if keypoints is not None:
                  vis_util.draw_keypoints_on_image_array(
                              image,
                              box_to_keypoints_map[box],
                              color=color,
                              radius=line_thickness / 2,
                              use_normalized_coordinates=use_normalized_coordinates)
      
      (firstquadrant,secondquadrant,thirdquadrant,fourthquadrant)=('O','O','O','O')
      for personbox in personboxes:
            ymin = personbox[0]*height
            xmin = personbox[1]*width
            ymax = personbox[2]*height
            xmax = personbox[3]*width            
            print(ymin,xmin,ymax,xmax)
            points=[(ymin,xmin),(ymax,xmin),(ymax,xmax),(ymin,xmax)]
            for point in points:
                  if point[0] < height/2  and point[1] < width/2 and firstquadrant=='O':
                        firstquadrant='L'
                  if point[0] < height/2  and point[1] > width/2 and secondquadrant=='O':
                        secondquadrant='L'
                  if point[0] > height/2  and point[1] < width/2 and thirdquadrant=='O':
                        thirdquadrant='L'
                  if point[0] > height/2  and point[1] > width/2 and fourthquadrant=='O':
                        fourthquadrant='L'
            
      print(firstquadrant," ",secondquadrant," ",thirdquadrant," ",fourthquadrant)
      urli="http://192.168.0.105/?led"+firstquadrant+secondquadrant+thirdquadrant+fourthquadrant
      print(urli)
      global quadcache
      if quadcache != firstquadrant+secondquadrant+thirdquadrant+fourthquadrant:
            try:
                  quadcache=firstquadrant+secondquadrant+thirdquadrant+fourthquadrant
                  t = threading.Thread(target=urlreq, args=(urli,))
                  t.start()
 
#            
            except:
                  print("thread not spawned")

#      print("personboxes=",personboxes)
      return image




MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017' #fast

#MODEL_NAME = 'faster_rcnn_resnet50_lowproposals_coco_2017_11_08' #fast 

#MODEL_NAME = 'ssd_inception_v2_coco_2017_11_08' #medium speed 
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

IMAGE_SIZE = (12, 8)

fileAlreadyExists = os.path.isfile(PATH_TO_CKPT)

if not fileAlreadyExists:
      print('Downloading frozen inference graph')
#      opener = urllib.request.URLopener()
      print('Downloading frozen inference graph')

#      opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
      tar_file = tarfile.open(MODEL_FILE)
      for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                  tar_file.extract(file, os.getcwd())
              
              
detection_graph = tf.Graph()
with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

camera = picamera.PiCamera()

camera.resolution = (width,height )
camera.vflip = True
camera.framerate = 30
rawCapture = PiRGBArray(camera, size = (width, height))
 



with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
            for frame in camera.capture_continuous(rawCapture, format="bgr"):        
                  image_np = np.array(frame.array)
                  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                  image_np_expanded = np.expand_dims(image_np, axis=0)
                  # Definite input and output Tensors for detection_graph 
                  image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                  # Each box represents a part of the image where a particular object was detected
                  detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                  # Each score represent how level of confidence for each of the objects.
                  # Score is shown on the result image, together with the class label.
                  detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                  detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                  num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                  print('Running detection..')
                  (boxes, scores, classes, num) = sess.run(
                  [detection_boxes, detection_scores, detection_classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})
                  print('Done.  Visualizing..')
#                  print('boxes=',boxes,' scores=', scores,"  classes=", classes,"  num=", num)
#                  print('boxes=',boxes.shape,' scores=', scores.shape,"  classes=", classes.shape,"  num=", num.shape)
                  i=0
                  cla=[]
#                  print(type(classes),classes.shape)
                  i=0
                  cla=[]
#                  print(type(classes),classes.shape)
                  for k in classes.reshape(classes.shape[1]):
                        i=i+1
                #        print(k)
                        if int(k)==1:
                              cla.append(i)
                  classes=classes[:,np.array(cla)-1]
                  boxes=boxes[:,np.array(cla)-1,:]
                  scores=scores[:,np.array(cla)-1]
#                  print('boxes=',boxes,' scores=', scores,"  classes=", classes,"  num=", num)
#                  print('boxes=',boxes.shape,' scores=', scores.shape,"  classes=", classes.shape,"  num=", num.shape)
                  visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
#                  print("after=",boxes.shape)
                  cv2.imshow('object detection', cv2.resize(image_np, (1280, 960)))
                  rawCapture.truncate(0)
                  if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
            print('exiting')
            cap.release()
            cv2.destroyAllWindows()