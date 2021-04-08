# Script to detect objects using Tensorflow 2 object detection API, crop person class
# Helper functions in detector.py
# Adopted from https://github.com/abdelrahman-gaber/tf2-object-detection-api-tutorial

import os
import cv2
import time
import argparse

from detector import DetectorTF2


def DetectImagesFromFolder(detector, images_dir, save_output=False, output_dir='output/'):

	for file in os.scandir(images_dir):
		if file.is_file() and file.name.endswith(('.jpg', '.jpeg', '.png')) :
			image_path = os.path.join(images_dir, file.name)
			print(image_path)
			img = cv2.imread(image_path)
			det_boxes = detector.DetectFromImage(img)

			filename = str(file.name)[:-4]
			# Call for cropping person and saving crop if save_output is True			
			detector.crop_person(img, det_boxes, filename, save_output)
			

			img = detector.DisplayDetections(img, det_boxes)

			cv2.imshow('TF2 Detection', img)
			cv2.waitKey(0)



if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Object Detection from Images')
	parser.add_argument('--model_path', help='Path to frozen detection model',
						default='models/efficientdet_d0_coco17_tpu-32/saved_model')
	parser.add_argument('--path_to_labelmap', help='Path to labelmap (.pbtxt) file',
	                    default='models/mscoco_label_map.pbtxt')
	parser.add_argument('--class_ids', help='id of classes to detect, expects string with ids delimited by ","',
	                    type=str, default=None) # example input "1,3" to detect person and car
	parser.add_argument('--threshold', help='Detection Threshold', type=float, default=0.4)
	parser.add_argument('--images_dir', help='Directory to input images)', default='images/detection/')

	parser.add_argument('--output_directory', help='Path to output images and video', default='images/crops/')

	parser.add_argument('--save_output', help='Flag for save images and video with detections visualized, default: False',
	                    action='store_true')  # default is false
	args = parser.parse_args()

	id_list = None
	if args.class_ids is not None:
		id_list = [int(item) for item in args.class_ids.split(',')]

	if args.save_output:
		if not os.path.exists(args.output_directory):
			os.makedirs(args.output_directory)

	# instance of the class DetectorTF2
	detector = DetectorTF2(args.model_path, args.path_to_labelmap, class_id=id_list, threshold=args.threshold)

	#if args.video_input:
	#	DetectFromVideo(detector, args.video_path, save_output=args.save_output, output_dir=args.output_directory)
	DetectImagesFromFolder(detector, args.images_dir, save_output=args.save_output, output_dir=args.output_directory)

	print("Done ...")
	cv2.destroyAllWindows()
