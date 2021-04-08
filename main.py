# Script to detect objects using Tensorflow 2 object detection API, crop person class and pass through pose estimator
# Helper functions in detector.py
# Adopted from https://github.com/abdelrahman-gaber/tf2-object-detection-api-tutorial

import os
import cv2
import time
import argparse

import math
import pickle
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import logging
import sys
import time

from detector import DetectorTF2

#from tf_pose import common
#from tf_pose.estimator import TfPoseEstimator
#from tf_pose.networks import get_graph_path, model_wh


def DetectImagesFromFolder(detector, images_dir, save_output=False, output_dir='output/', show_pose=False):

	for file in os.scandir(images_dir):
		if file.is_file() and file.name.endswith(('.jpg', '.jpeg', '.png')) :
			image_path = os.path.join(images_dir, file.name)
			print(image_path)
			img = cv2.imread(image_path)
			det_boxes = detector.DetectFromImage(img)

			filename = str(file.name)[:-4]
			# Call for cropping person, returns list of persons detected			
			persons = detector.crop_person(img, det_boxes)
			person_bbox = detector.get_person_bbox(img, det_boxes)
			print(person_bbox)
			
			# Save cropped person if save_output is True
			if save_output:
				person_count = 0
				for person in persons:
					cv2.imwrite("images/detected_crops/"+str(filename)+"_"+str(person_count)+".jpg", person)
					person_count+=1

			if show_pose:
				for person in persons:
					pass

			img = detector.DisplayDetections(img, det_boxes)

			cv2.imshow('TF2 Detection', img)
			cv2.waitKey(0)
			#break

'''logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)'''


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Object Detection from Images')
	
	parser.add_argument('--model_path_detector', help='Path to frozen detection model',
						default='models/efficientdet_d0_coco17_tpu-32/saved_model')

	parser.add_argument('--path_to_labelmap', help='Path to labelmap (.pbtxt) file',
	                    default='models/mscoco_label_map.pbtxt')
	
	parser.add_argument('--class_ids', help='id of classes to detect, expects string with ids delimited by ","',
	                    type=str, default=None) # example input "1,3" to detect person and car
	
	parser.add_argument('--threshold', help='Detection Threshold', type=float, default=0.4)
	
	parser.add_argument('--images_dir', help='Directory to input images)', default='images/detection/')

	parser.add_argument('--output_directory', help='Path to output images and video', default='images/detected_crops/')

	parser.add_argument('--save_output', help='Flag for save images and video with detections visualized, default: False',
	                    action='store_true')  # default is false
	
	parser.add_argument('--show_pose', help = 'Flag to show pose estimation for cropped images, default: False',
	 action ='store_true')

	parser.add_argument('--model_pose', help='Pose model: cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small',
						default='mobilenet_thin')
	parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
	args = parser.parse_args()

	id_list = None
	if args.class_ids is not None:
		id_list = [int(item) for item in args.class_ids.split(',')]

	if args.save_output:
		if not os.path.exists(args.output_directory):
			os.makedirs(args.output_directory)

	# instance of the class DetectorTF2
	detector = DetectorTF2(args.model_path_detector, args.path_to_labelmap, class_id=id_list, threshold=args.threshold)

	#if args.video_input:
	#	DetectFromVideo(detector, args.video_path, save_output=args.save_output, output_dir=args.output_directory)
	DetectImagesFromFolder(detector, args.images_dir, save_output=args.save_output, output_dir=args.output_directory, show_pose=args.show_pose)


	print("Done ...")
	cv2.destroyAllWindows()
