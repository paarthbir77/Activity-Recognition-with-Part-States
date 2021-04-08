import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import cv2 as cv
import math

model_path = "models/posenet/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
template_path = "images/crops/70.jpg"
target_path = "images/crops/70.jpg"

# Load TFLite model and allocate tensors (memory usage method reducing latency)
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors information from the model file
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

template_image_src = cv.imread(template_path)
# src_tepml_width, src_templ_height, _ = template_image_src.shape 
template_image = cv.resize(template_image_src, (width, height))
cv.imshow('win',template_image)
cv.waitKey(0)
cv.destroyAllWindows()

target_image_src = cv.imread(target_path)
# src_tar_width, src_tar_height, _ = target_image_src.shape 
target_image = cv.resize(target_image_src, (width, height))
cv.imshow('win',target_image)
cv.waitKey(0)
cv.destroyAllWindows()

# add a new dimension to match model's input
template_input = np.expand_dims(template_image.copy(), axis=0)
target_input = np.expand_dims(target_image.copy(), axis=0)

# check the type of the input tensor
floating_model = input_details[0]['dtype'] == np.float32

if floating_model:
  template_input = (np.float32(template_input) - 127.5) / 127.5
  target_input = (np.float32(target_input) - 127.5) / 127.5

# Process template image
# Sets the value of the input tensor
interpreter.set_tensor(input_details[0]['index'], template_input)
# Runs the computation
interpreter.invoke()
# Extract output data from the interpreter
template_output_data = interpreter.get_tensor(output_details[0]['index'])
template_offset_data = interpreter.get_tensor(output_details[1]['index'])
# Getting rid of the extra dimension
template_heatmaps = np.squeeze(template_output_data)
template_offsets = np.squeeze(template_offset_data)
print("template_heatmaps' shape:", template_heatmaps.shape)
print("template_offsets' shape:", template_offsets.shape)

# Process target image. Same commands
interpreter.set_tensor(input_details[0]['index'], target_input)
interpreter.invoke()
target_output_data = interpreter.get_tensor(output_details[0]['index'])
target_offset_data = interpreter.get_tensor(output_details[1]['index'])
target_heatmaps = np.squeeze(target_output_data)
target_offsets = np.squeeze(target_offset_data)

def parse_output(heatmap_data,offset_data, threshold):

  '''
  Input:
    heatmap_data - hetmaps for an image. Three dimension array
    offset_data - offset vectors for an image. Three dimension array
    threshold - probability threshold for the keypoints. Scalar value
  Output:
    array with coordinates of the keypoints and flags for those that have
    low probability
  '''

  joint_num = heatmap_data.shape[-1]
  pose_kps = np.zeros((joint_num,3), np.uint32)

  for i in range(heatmap_data.shape[-1]):

      joint_heatmap = heatmap_data[...,i]
      max_val_pos = np.squeeze(np.argwhere(joint_heatmap==np.max(joint_heatmap)))
      remap_pos = np.array(max_val_pos/8*257,dtype=np.int32)
      pose_kps[i,0] = int(remap_pos[0] + offset_data[max_val_pos[0],max_val_pos[1],i])
      pose_kps[i,1] = int(remap_pos[1] + offset_data[max_val_pos[0],max_val_pos[1],i+joint_num])
      max_prob = np.max(joint_heatmap)

      if max_prob > threshold:
        if pose_kps[i,0] < 257 and pose_kps[i,1] < 257:
          pose_kps[i,2] = 1

  return pose_kps

def draw_kps(show_img,kps, ratio=None):
    for i in range(0,kps.shape[0]):
      if kps[i,2]:#kps[i,2]
        if isinstance(ratio, tuple):
          cv.circle(show_img,(int(round(kps[i,1]*ratio[1])),int(round(kps[i,0]*ratio[0]))),2,(0,255,255),round(int(1*ratio[1])))
          print(int(round(kps[i,1]*ratio[1])),int(round(kps[i,0]*ratio[0])))
          continue
        cv.circle(show_img,(kps[i,1],kps[i,0]),2,(0,255,255),-1)
        print(kps[i,1],kps[i,0])
    return show_img

template_show = np.squeeze((template_input.copy()*127.5+127.5)/255.0)
template_show = np.array(template_show*255,np.uint8)
template_kps = parse_output(template_heatmaps,template_offsets,0.3)
print(template_kps)
cv.imshow('win',draw_kps(template_show.copy(),template_kps))
cv.waitKey(0)
cv.destroyAllWindows()

#target_show = np.squeeze((target_input.copy()*127.5+127.5)/255.0)
#target_show = np.array(target_show*255,np.uint8)
#target_kps = parse_output(target_heatmaps,target_offsets,0.3)
#cv.imshow('win',draw_kps(target_show.copy(),target_kps))
#cv.waitKey(0)
#cv.destroyAllWindows()
