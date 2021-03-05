#!/usr/bin/env python3
import sys
import os.path
import orbslam2




from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo
import fcntl

import numpy as np
import cv2 as cv
import traceback
import g2o
from threading import Thread
import os
import shutil

from sptam.dynaseg import DynaSeg
from sptam.msptam import SPTAM, stereoCamera
from sptam.components import Camera
from sptam.components import StereoFrame
from sptam.feature import ImageFeature
from sptam.params import ParamsKITTI
from sptam.dataset import KITTIOdometry

import time

def main(orb_path, device, data_path, save, sequence):
    sequence_path = os.path.join(data_path, sequence)
    vocab_path = os.path.join(orb_path,'Vocabulary/ORBvoc.txt')
    ins = int(sequence)
    if ins < 3:
        settings_path = os.path.join(orb_path,'Examples/Stereo/KITTI00-02.yaml')
    elif ins == 3:
        settings_path = os.path.join(orb_path, 'Examples/Stereo/KITTI03.yaml')
    else:
        settings_path = os.path.join(orb_path, 'Examples/Stereo/KITTI04-12.yaml')
    print(settings_path)

    coco_path = '../../maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml'
    params = ParamsKITTI()
    dataset = KITTIOdometry(sequence_path)
    disp_path = os.path.join('/usr/stud/linp/storage/user/linp/disparity/',sequence)

    feature_params = dict(maxCorners=1000,
                          qualityLevel=0.1,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

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

    sptam = SPTAM(params)

    config = stereoCamera()
    mtx = np.array([[707.0912, 0, 601.8873], [0, 707.0912, 183.1104], [0, 0, 1]])
    dist = np.array([[0] * 4]).reshape(1, 4).astype(np.float32)

    dilation = 2
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))

    cam = Camera(
        dataset.cam.fx, dataset.cam.fy, dataset.cam.cx, dataset.cam.cy,
        dataset.cam.width, dataset.cam.height,
        params.frustum_near, params.frustum_far,
        dataset.cam.baseline)

    num_images = len(dataset)

    slam = orbslam2.System(vocab_path, settings_path, orbslam2.Sensor.STEREO)
    slam.set_use_viewer(False)
    slam.initialize()

    times_track = [0 for _ in range(num_images)]
    times = [0 for _ in range(num_images)]
    print('-----')
    print('Start processing sequence {}'.format(sequence))
    print('Images in the sequence: {0}'.format(num_images))

    loadmodel = './finetune_300.tar'
    config_file = coco_path
    # "configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", device])
    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
    )

    if save == '1':
        path = './{}'.format(sequence)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

    iml = cv.imread(dataset.left[0], cv.IMREAD_UNCHANGED)
    dseg = DynaSeg(iml, coco_demo, feature_params, disp_path, config, paraml, lk_params, mtx, dist, kernel,loadmodel)
    for idx in range(num_images):
        t0 = time.time()
        left_image = cv.imread(dataset.left[idx], cv.IMREAD_UNCHANGED)
        right_image = cv.imread(dataset.right[idx], cv.IMREAD_UNCHANGED)
        # original
        featurel = ImageFeature(left_image, params)
        featurer = ImageFeature(right_image, params)
        timestamp = dataset.timestamps[idx]

        t = Thread(target=featurer.extract)
        t.start()
        featurel.extract()
        t.join()

        print('{}. frame'.format(idx))
        try:
            frame = StereoFrame(idx, g2o.Isometry3d(), featurel, featurer, cam, timestamp=timestamp)

            if not sptam.is_initialized():
                sptam.initialize(frame)
            else:
                sptam.track(frame)

            if idx % 5 == 0:
                if idx:
                    c = dseg.dyn_seg_rec(frame, left_image, idx)
                dseg.updata(left_image, right_image, idx, frame)
            else:
                c = dseg.dyn_seg_rec(frame, left_image, idx)
            if idx:
                c = cv.dilate(c, kernel)
                left_mask = c.reshape(dseg.h,dseg.w,1)
                right_mask = c.reshape(dseg.h,dseg.w,1)
                if save == '1':
                    cv.imwrite('./{}/{}.png'.format(sequence,idx), c*255)
            else:
                left_mask = np.ones((dseg.h,dseg.w,1),dtype=np.uint8)
                right_mask = np.ones((dseg.h,dseg.w,1),dtype=np.uint8)
            #
            if left_image is None:
                print("failed to load image at {0}".format(dataset.left[idx]))
                return 1
            if right_image is None:
                print("failed to load image at {0}".format(dataset.right[idx]))
                return 1

            t1 = time.time()
            slam.process_image_stereo(left_image[:, :, ::-1], right_image[:, :, ::-1], left_mask, right_mask, timestamp)
            t2 = time.time()

            ttrack = t2 - t1
            times_track[idx] = ttrack
            times[idx] = t2 - t0

            t = 0
            if idx < num_images - 1:
                t = dataset.timestamps[idx + 1] - timestamp
            elif idx > 0:
                t = timestamp - dataset.timestamps[idx - 1]

            if ttrack < t:
                time.sleep(t - ttrack)
        except:
            traceback.print_exc()
            print('error in frame {}'.format(idx))
            break
    i = 0
    result_path = 'ro/s{}{}.txt'.format(sequence, i)
    while True:
        if not os.path.exists(result_path):
            s_flag = save_trajectory(slam.get_trajectory_points(), result_path)
            if s_flag:
                print(result_path)
                break
        i += 1
        result_path = 'ro/s{}{}.txt'.format(sequence, i)

    slam.shutdown()
    sptam.stop()
    times_track = sorted(times_track)
    total_time = sum(times_track)
    ttime = sum(times)
    print('-----')
    print('mean ORB tracking time: {0}'.format(total_time / num_images))
    print('mean DSR tracking time: {0}'.format(ttime / num_images))
    print('mean dcverror: {}'.format(np.mean(dseg.cverrs)))

    return 0



def save_trajectory(trajectory, filename):
    try:
        with open(filename, 'w') as traj_file:
            fcntl.flock(traj_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            traj_file.writelines('{r00} {r01} {r02} {t0} {r10} {r11} {r12} {t1} {r20} {r21} {r22} {t2}\n'.format(
                r00=repr(r00),
                r01=repr(r01),
                r02=repr(r02),
                t0=repr(t0),
                r10=repr(r10),
                r11=repr(r11),
                r12=repr(r12),
                t1=repr(t1),
                r20=repr(r20),
                r21=repr(r21),
                r22=repr(r22),
                t2=repr(t2)
            ) for stamp, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 in trajectory)
        traj_file.close()
        return 1
    except:
        return 0

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print('Usage: ./orbslam_stereo_kitti path_to_orb device path_to_data save_img sequence ')
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
