from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo

from RDRSeg import RDRSeg
from sptam.msptam import stereoCamera

import orbslam2

import sys
import cv2 as cv
import traceback
import numpy as np
import os
import shutil
import time

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

sequence_path = os.path.join('/storage/remote/atcremers17/linp/dataset/kittic/sequences/',sequence)
file_path = os.path.join(sequence_path, 'image_2')
left_filenames = load_images(file_path)
file_path = os.path.join(sequence_path, 'image_3')
right_filenames = load_images(file_path)
timestamps = load_times(sequence_path)

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
f = 0
config = stereoCamera(sequence)
num_images = len(left_filenames)

rdrseg = RDRSeg(iml, coco_demo, depth_path, kernel, config)

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

dpath = 'pmask/{}{}/'.format(mode,sequence) #
if os.path.exists(dpath):
    shutil.rmtree(dpath)
os.mkdir(dpath)

start_time = time.time()
for idx in range(num_images):
    print('{} frame'.format(idx))
    left_image = cv.imread(left_filenames[idx], cv.IMREAD_UNCHANGED)
    right_image = cv.imread(right_filenames[idx], cv.IMREAD_UNCHANGED)
    prob_image = cv.imread(prob_filenames[idx])
    timestamp = timestamps[idx]

    left_mask = np.ones((rdrseg.h, rdrseg.w, 1), dtype=np.uint8)
    right_mask = np.ones((rdrseg.h, rdrseg.w, 1), dtype=np.uint8)
    slam0.process_image_stereo(left_image[:, :, ::-1], right_image[:, :, ::-1], left_mask, right_mask, timestamp)
    trans = pose_to_transformation(slam0.get_trajectory_points()[-1])
    if idx % 3 == 0:
        if idx:
            rdrseg.updata(left_image, right_image, idx, trans)
    c = rdrseg.rdr_seg_rec(left_image, prob_image, idx,trans)
    cv.imwrite(os.path.join(dpath, '{0:06}.png'.format(idx)), c)
mean_time = (time.time() - start_time) / num_images
print('sequence ',sequence)
print('mean process time: {}'.format(round(mean_time,2)))