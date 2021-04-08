import argparse
import logging
import sys
import time

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

def crop_and_save(id, dimensions, image):
	if id == 0:
		path = "/home/prth/Desktop/pose_estimate/tf-pose-estimation/images/part_crops/head/0.jpg"
	elif id == 1 or id == 2:
		path = "a"
	elif id == 3 or id == 4:
		path = "b"
	#cv2.imwrite(path, image[dimensions[1]-dimensions[2]:dimensions[1]+dimensions[2], dimensions[0]-dimensions[2]:dimensions[0]+dimensions[2] ])
	cv2.imshow('crop', image[dimensions[1]-dimensions[2]:dimensions[1]+dimensions[2], dimensions[0]-dimensions[2]:dimensions[0]+dimensions[2]])
	cv2.waitKey(0)
	cv2.destroyAllWindows()


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

    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)
    if image is None:
        logger.error('Image can not be read, path=%s' % args.image)
        sys.exit(-1)

    t = time.time()
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))
    # look in this function
    image, part_list = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    print("main script")
    print(part_list)
    # Function call to crop body parts
    part_bbox = crop_parts.crop_parts(part_list)
    # Make bounding boxes
    # To implement: if box exceeds image dimensions
    image2 = common.read_imgfile(args.image, None, None)
    image_orig = common.read_imgfile(args.image, None, None)
    for i in range(len(part_bbox)):
        if part_bbox[i][1][0] != -1:
        	image2 = cv2.rectangle(image, (part_bbox[i][1][0]+part_bbox[i][1][2],part_bbox[i][1][1]+part_bbox[i][1][2]), (part_bbox[i][1][0]-part_bbox[i][1][2],part_bbox[i][1][1]-part_bbox[i][1][2]), (0,255,0),1)
    cv2.imshow('result cv2', image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(part_bbox)
    
    for i in range(len(args.save_crop)):
    	if args.save_crop[i]=='1':
    		if part_bbox[i][1][0]!=-1:
    			print("save crop of ", i, part_bbox[i][1])
    			crop_and_save(i, part_bbox[i][1], image_orig)

    		else:
    			print("no crop available for", i)
    		

    """try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        a = fig.add_subplot(2, 2, 1)
        a.set_title('Result')
        #for i in range(len(part_bbox)):
        #	if part_bbox[i][0] != -1:
        #		image2 = cv2.Rectangle(image, (part_bbox[i][0]+part_bbox[i][2],part_bbox[i][1]+part_bbox[i][2]), 
        #			(part_bbox[i][0]-part_bbox[i][2],part_bbox[i][1]-part_bbox[i][2]))
        #cv2.imshow('result cv2', image2)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

        # show network output
        a = fig.add_subplot(2, 2, 2)
        plt.imshow(bgimg, alpha=0.5)
        tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        tmp2 = e.pafMat.transpose((2, 0, 1))
        tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
        tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

        a = fig.add_subplot(2, 2, 3)
        a.set_title('Vectormap-x')
        # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        a = fig.add_subplot(2, 2, 4)
        a.set_title('Vectormap-y')
        # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()
        plt.show()"""
    """except Exception as e:
        logger.warning('matplitlib error, %s' % e)
        cv2.imshow('result', image)
        cv2.waitKey()"""
