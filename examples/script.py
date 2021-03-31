# a = 0.353772
# b = 0.324035
# c = (a+b) / 2
# import random
# for _ in range(3):
#     print(c+(a-b)/2*(random.randrange(1000)-500)/500)

# import pandas as pd
# a = [2.378798129,2.379137925,2.379728386,2.378665383,2.380813182,2.379428472,2.378278658,2.37854379,2.37832403,2.377241082,2.378306558,2.378138779]
# b = [0.14646,0.14865,0.145923,0.137563,0.145271,0.144723032,0.142944,0.131826,0.147774,0.137352,0.144257,0.140717264]
#
# data = pd.DataFrame({'a':a,'b':b})
# print(data.corr())

# import fcntl
# f = open('./test.txt','w')
# for i in range(10):
#     f.write(str(i))
# fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
# try:
#     f0 = open('./test.txt','a')
#     f0.write('he')
#     fcntl.flock(f0, fcntl.LOCK_EX|fcntl.LOCK_NB)
#     f0.write('hello')
# except:
#     print('succesfully!')
#     f0.close()
# finally:
#     f.close()
# f0 = open('./test.txt','a')
# fcntl.flock(f0, fcntl.LOCK_EX|fcntl.LOCK_NB)
# f0.write('world!')
# f0.close()

from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo

from PDSeg import PDSeg
from sptam.dynaseg import DynaSegt
from sptam.msptam import stereoCamera

import orbslam2

import sys
import cv2 as cv
import traceback
import numpy as np
import os
import shutil

def load_images(path_to_sequence):
    res = [os.path.join(path_to_sequence, img) for img in os.listdir(path_to_sequence)]
    res.sort()
    return res

def load_times(path_to_sequence):
    timestamps = []
    with open(os.path.join(path_to_sequence, 'times.txt')) as times_file:
        for line in times_file:
            if len(line) > 0:
                timestamps.append(float(line))
    return timestamps

def pose_to_transformation(pose):
    res = np.zeros((4,4))
    for i in range(3):
        res[i,:3] = pose[4*i+1:4*(i+1)]
        res[i,3] = pose[4*i]
    res[3,3] = 1
    res = np.linalg.inv(res)
    return res

sequence = sys.argv[1]
mode = sys.argv[2]

file_path = os.path.join('/storage/remote/atcremers17/linp/dataset/kittic/sequences/',sequence, 'image_2')
left_filenames = load_images(file_path)
file_path = os.path.join('/storage/remote/atcremers17/linp/dataset/kittic/sequences/',sequence, 'image_3')
right_filenames = load_images(file_path)
timestamps = load_times(os.path.join('/storage/remote/atcremers17/linp/dataset/kittic/sequences/',sequence))

prob_path = os.path.join('/usr/stud/linp/storage/user/linp/prob/', sequence)
prob_filenames = load_images(prob_path)

config_file = '../../maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml'
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", 'cuda'])
coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

dilation = 2
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))

depth_path = os.path.join('/usr/stud/linp/storage/user/linp/depth/',sequence)

iml = cv.imread(left_filenames[0], cv.IMREAD_UNCHANGED)
if mode == 'dpr' or mode == 'tt':
    pdseg = PDSeg(iml,coco_demo,depth_path,kernel)
else:
    orb_path = '/usr/stud/linp/storage/user/linp/ORB_SLAM2'
    vocab_path = os.path.join(orb_path, 'Vocabulary/ORBvoc.txt')
    ins = int(sequence)
    if ins < 3:
        settings_path = os.path.join(orb_path, 'Examples/Stereo/KITTI00-02.yaml')
    elif ins == 3:
        settings_path = os.path.join(orb_path, 'Examples/Stereo/KITTI03.yaml')
    else:
        settings_path = os.path.join(orb_path, 'Examples/Stereo/KITTI04-12.yaml')
    slam0 = orbslam2.System(vocab_path, settings_path, orbslam2.Sensor.STEREO)
    slam0.set_use_viewer(False)
    slam0.initialize()
    feature_params = dict(maxCorners=1000,
                          qualityLevel=0.1,
                          minDistance=7,
                          blockSize=7)
    config = stereoCamera()
    mtx = np.array([[707.0912, 0, 601.8873], [0, 707.0912, 183.1104], [0, 0, 1]])
    dist = np.array([[0] * 4]).reshape(1, 4).astype(np.float32)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    loadmodel = './finetune_300.tar'
    paraml = {'minDisparity': 1,
              'numDisparities': 64,
              'blockSize': 10,
              'P1': 4 * 3 * 9 ** 2,
              'P2': 4 * 3 * 9 ** 2,
              'disp12MaxDiff': 1,
              'preFilterCap': 10,
              'uniquenessRatio': 15,
              'speckleWindowSize': 100,
              'speckleRange': 1,
              'mode': cv.STEREO_SGBM_MODE_SGBM_3WAY
              }

    dseg = DynaSegt(iml, coco_demo, feature_params, depth_path, config, paraml, lk_params, mtx, dist, kernel, loadmodel)


num_images = len(left_filenames)

dpath = 'pmask/{}{}/'.format(mode,sequence)
if os.path.exists(dpath):
    shutil.rmtree(dpath)
os.mkdir(dpath)

print('sequence ',sequence)
for idx in range(num_images):
    left_image = cv.imread(left_filenames[idx], cv.IMREAD_UNCHANGED)
    right_image = cv.imread(right_filenames[idx], cv.IMREAD_UNCHANGED)
    prob_image = cv.imread(prob_filenames[idx])
    # dpr
    if mode == 'dpr':
        c = pdseg.pd_seg_rec(left_image, prob_image,idx)
        cv.imwrite(os.path.join(dpath, '{0:06}.png'.format(idx)), c*255)
    # t
    elif mode == 'tt':
        c = pdseg.pd_seg_t(left_image, prob_image)
        cv.imwrite(os.path.join(dpath, '{0:06}.png'.format(idx)), c)
    else:
        left_mask = np.ones((dseg.h, dseg.w, 1), dtype=np.uint8)
        right_mask = np.ones((dseg.h, dseg.w, 1), dtype=np.uint8)
        timestamp = timestamps[idx]
        slam0.process_image_stereo(left_image[:, :, ::-1], right_image[:, :, ::-1], left_mask, right_mask, timestamp)
        trans = pose_to_transformation(slam0.get_trajectory_points()[-1])
        if idx % 3 == 0:
            if idx:
                c = dseg.dyn_seg_rec(trans, left_image, idx)
            dseg.updata(left_image, right_image, idx, trans)
        else:
            c = dseg.dyn_seg_rec(trans, left_image, idx)
        if idx:
            cv.imwrite(os.path.join(dpath, '{0:06}.png'.format(idx)), c * 255)
    print('{} frame'.format(idx))
