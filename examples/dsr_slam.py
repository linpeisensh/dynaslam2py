#!/usr/bin/env python3
import sys
import os.path
import orbslam2




from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo

import numpy as np
import cv2 as cv
import time
import traceback
import os
import shutil

from sptam.dynaseg import DynaSeg
from sptam.msptam import stereoCamera
from sptam.params import ParamsKITTI
from sptam.dataset import KITTIOdometry


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


    config = stereoCamera()
    mtx = np.array([[707.0912, 0, 601.8873], [0, 707.0912, 183.1104], [0, 0, 1]])
    dist = np.array([[0] * 4]).reshape(1, 4).astype(np.float32)

    dilation = 2

    num_images = len(dataset)

    slam = orbslam2.System(vocab_path, settings_path, orbslam2.Sensor.STEREO)
    slam.set_use_viewer(False)
    slam.initialize()

    slam0 = orbslam2.System(vocab_path, settings_path, orbslam2.Sensor.STEREO)
    slam0.set_use_viewer(False)
    slam0.initialize()

    if save == '1':
        path = './dym'
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

    times_track = [0 for _ in range(num_images)]
    print('-----')
    print('Start processing sequence {}'.format(sequence))
    print('Images in the sequence: {0}'.format(num_images))

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

    iml = cv.imread(dataset.left[0], cv.IMREAD_UNCHANGED)
    dseg = DynaSeg(iml, coco_demo, feature_params, disp_path, config, paraml, lk_params, mtx, dist, dilation)
    for idx in range(num_images):
        left_image = cv.imread(dataset.left[idx], cv.IMREAD_UNCHANGED)
        right_image = cv.imread(dataset.right[idx], cv.IMREAD_UNCHANGED)
        timestamp = dataset.timestamps[idx]

        print('{}. frame'.format(idx))
        try:
            left_mask = np.ones((dseg.h, dseg.w, 1), dtype=np.uint8)
            right_mask = np.ones((dseg.h, dseg.w, 1), dtype=np.uint8)
            slam0.process_image_stereo(left_image[:, :, ::-1], right_image[:, :, ::-1], left_mask, right_mask, timestamp)
            transformation = pose_to_transformation(slam0.get_trajectory_points()[-1])


            if idx % 5 == 0:
                if idx:
                    c = dseg.dyn_seg_rec(transformation, left_image, idx)
                dseg.updata(left_image, right_image, idx, transformation)
            else:
                c = dseg.dyn_seg_rec(transformation, left_image, idx)
            if idx:
                left_mask = c.reshape(dseg.h,dseg.w,1)
                right_mask = c.reshape(dseg.h,dseg.w,1)
                if save == '1':
                    cv.imwrite('./dym/{}.png'.format(idx), c*255)

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
    save_trajectory(slam.get_trajectory_points(), '../../results/kitti/a{}.txt'.format(sequence))

    slam.shutdown()
    slam0.shutdown()
    times_track = sorted(times_track)
    total_time = sum(times_track)
    print('-----')
    print('median tracking time: {0}'.format(times_track[num_images // 2]))
    print('mean tracking time: {0}'.format(total_time / num_images))

    return 0

def pose_to_transformation(pose):
    res = np.zeros((4,4))
    for i in range(3):
        res[i,:3] = pose[4*i+1:4*(i+1)]
        res[i,3] = pose[4*i]
    res[3,3] = 1
    res = np.linalg.inv(res)
    return res

def save_trajectory(trajectory, filename):
    with open(filename, 'w') as traj_file:
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


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print('Usage: ./orbslam_stereo_kitti path_to_orb device path_to_data save sequence ')
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
