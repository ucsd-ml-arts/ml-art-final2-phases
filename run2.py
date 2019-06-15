import argparse
import logging
import sys
import time
import os

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import re
def sorted_nicely( l ):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    # 1. CHANGED IMAGE FILE TO DIRECTORY NAME WITH IMAGES
    # parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--directory', type=str, default='./images')
    parser.add_argument('--model', type=str, default='mobilenet_thin',
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


# 2. ITERATE OVER EVERY IMAGE IN DIRECTORY
    num_imgs = len([name for name in os.listdir(args.directory) if os.path.isfile(os.path.join(args.directory, name))])
    print("num_imgs: %d" %num_imgs)

    filelist = sorted_nicely(os.listdir(args.directory))
    for filename in filelist:
        if filename.endswith(".jpg") or filename.endswith(".png"):

             # print(os.path.join(directory, filename))
             # estimate human poses from a single image !
             img_path = os.path.join(args.directory, filename)

             # logger.debug("filename %s" %filename)
             image = common.read_imgfile(img_path, None, None)
             if image is None:
                 logger.error('Image can not be read, path=%s' % img_path)
                 sys.exit(-1)

             t = time.time()
             humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
             elapsed = time.time() - t

             logger.info('inference image: %s in %.4f seconds.' % (img_path, elapsed))

             logger.info('humans: %d' %len(humans))
             image = TfPoseEstimator.draw_humans(image, humans, img_name = filename, imgcopy=False)

             # To avoid showing in matplotlib
             continue

             try:
                 import matplotlib.pyplot as plt

                 fig = plt.figure()
                 a = fig.add_subplot(2, 2, 1)
                 a.set_title('Result')
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
                 plt.show()
             except Exception as e:
                 logger.warning('matplitlib error, %s' % e)
                 cv2.imshow('result', image)
                 cv2.waitKey()



        #     continue
        # else:
        #     continue
