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

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
    
# parts = ['nose', 'neck', 'L Shoulder', 'L Elbow', 'L Wrist', 'R Shoulder', 'R Elbow', 'R Wrist', 'L Hip', 'L Knee', 'L Ankle','R Hip', 'R Knee', 'R Ankle', 'L Eye', 'R Eye', 'L ear', 'R ear']

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

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    images_dir="/home/prth/Desktop/pose_estimate/tf-pose-estimation/images/crops"
    df = pd.DataFrame(columns = ['filename', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10','11', '12', '13', '14', '15', '16', '17'])

    # estimate human poses from a single image !
    for file in os.scandir(images_dir):
        image_path = os.path.join(images_dir, file.name)
        image = cv2.imread(image_path)
    #image = common.read_imgfile(args.image, None, None)
        if image is None:
            logger.error('Image can not be read, path=%s' % args.image)
            sys.exit(-1)

        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        elapsed = time.time() - t

        logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))
    # look in this function
        image, part_list = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        #print("main script")
        print(part_list)
        row = [(None, None)]*18
        for i in range(len(part_list)):
            row[int(part_list[i][0])] = (part_list[i][1], part_list[i][2])
            #df_new = pd.DataFrame()
        row.insert(0, str(image_path))
        df.loc[len(df)] = row

    df.to_csv('parts.csv')