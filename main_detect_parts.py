import argparse
import logging
import sys
import time
import os
import pandas as pd
from tf_pose import common
from tf_pose import crop_parts
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tf_pose import person_crop_list
from tf_pose import crop_parts
from tf_pose import bbox_kpt_match

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
    
# parts = ['nose', 'neck', 'L Shoulder', 'L Elbow', 'L Wrist', 'R Shoulder', 'R Elbow', 'R Wrist', 'L Hip', 'L Knee', 'L Ankle','R Hip', 'R Knee', 'R Ankle', 'L Eye', 'R Eye', 'L ear', 'R ear']

def crop_and_save(id, dimensions, image, name):
    if id == 0:
        path = "/home/prth/Desktop/pose_estimate/tf-pose-estimation/images/part_crops/head-test/"+str(name)+".jpg"
    elif id == 1 or id == 2:
        path = "a"
    elif id == 3 or id == 4:
        path = "b"
    
    y_min = dimensions[1]-dimensions[2]
    if y_min<0:
        y_min = 0
    y_max = dimensions[1]+dimensions[2]
    
    if y_max>image.shape[0]:
        y_max = image.shape[0]

    x_min = dimensions[0]-dimensions[2]
    if x_min<0:
        x_min = 0
    x_max = dimensions[0]+dimensions[2]
    if x_max>image.shape[1]:
        x_max = image.shape[1]
    cv2.imwrite(path, image[y_min:y_max, x_min:x_max])
    #cv2.imshow('crop', image[y_min:y_max, x_min:x_max])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--save_crop', type=str, default='00000',
                        help='for saving crop of body id i set string[i] to 1')

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    #images_dir="/home/prth/Desktop/pose_estimate/tf-pose-estimation/images/crops"
    #images_dir= "/home/prth/Downloads/WIDER_train/images/13--Interview"
    images_dir= "/home/prth/Desktop/pose_estimate/tf-pose-estimation/images/detection"
    
    #images_dir= "/home/prth/Downloads/WIDER_test/images/50--Celebration_Or_Party"  
    #set to 0
    count = 614

    for file in os.scandir(images_dir):
        image_path = os.path.join(images_dir, file.name)
        image = cv2.imread(image_path)
        image_orig = cv2.imread(image_path)
    #image = common.read_imgfile(args.image, None, None)
        if image is None:
            logger.error('Image can not be read, path=%s' % args.image)
            sys.exit(-1)

        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        elapsed = time.time() - t

        logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))
        #cv2.imshow('orig', image)
        #cv2.waitKey(0)

    # look in this function    
        image, part_list = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        #print("main script")
        print(part_list)
        persons_part_list = person_crop_list.get_unique_persons_from_kpt_list(part_list)
        print(persons_part_list)
        part_bbox = []
        for i in range(len(persons_part_list)):
            part_bbox.append(crop_parts.crop_parts(persons_part_list[i]))
        #print(part_bbox)
        #break
        #part_bbox = crop_parts.crop_parts(part_list)
        # Make bounding boxes
        # To implement: if box exceeds image dimensions
        
        image2 = image
        for i in range(len(part_bbox)):
            for j in range(len(part_bbox[i])):
                if part_bbox[i][j][0] != -1:
                    image2 = cv2.rectangle(image, (part_bbox[i][j][1][0]+part_bbox[i][j][1][2],part_bbox[i][j][1][1]+part_bbox[i][j][1][2]), (part_bbox[i][j][1][0]-part_bbox[i][j][1][2],part_bbox[i][j][1][1]-part_bbox[i][j][1][2]), (0,255,0),1)
        
        cv2.imshow('result cv2', image2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        for i in range(len(args.save_crop)):
            if args.save_crop[i]=='1':
                if part_bbox[i][1][0]!=-1:
                    print("save crop of ", i, part_bbox[i][1], count)
                    crop_and_save(i, part_bbox[i][1], image_orig,count)
                    count+=1

                else:
                    print("no crop available for", i)
        
