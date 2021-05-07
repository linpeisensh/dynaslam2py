from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo

from RDTSeg import RDTSeg
from sptam.msptam import stereoCamera

import orbslam2

import sys
import cv2 as cv
import fcntl
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

def main(orb_path, data_path, device, save, sequence):
    sequence_path = os.path.join(data_path, sequence)
    vocab_path = os.path.join(orb_path, 'Vocabulary/ORBvoc.txt')
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
    cfg.merge_from_list(["MODEL.DEVICE", device])
    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
    )

    dilation = 2
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))

    depth_path = os.path.join('/usr/stud/linp/storage/user/linp/depth/',sequence)

    iml = cv.imread(left_filenames[0], cv.IMREAD_UNCHANGED)
    config = stereoCamera(sequence)
    num_images = len(left_filenames)

    rdtseg = RDTSeg(iml, coco_demo, depth_path, kernel, config)

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

    slam = orbslam2.System(vocab_path, settings_path, orbslam2.Sensor.STEREO)
    slam.set_use_viewer(False)
    slam.initialize()

    if save == '1':
        dpath = 'pmask/rdr{}/'.format(sequence) #
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

        left_mask = np.ones((rdtseg.h, rdtseg.w, 1), dtype=np.uint8)
        right_mask = np.ones((rdtseg.h, rdtseg.w, 1), dtype=np.uint8)
        slam0.process_image_stereo(left_image[:, :, ::-1], right_image[:, :, ::-1], left_mask, right_mask, timestamp)
        trans = pose_to_transformation(slam0.get_trajectory_points()[-1])
        if idx % 3 == 0:
            if idx:
                rdtseg.update(left_image, right_image, idx, trans)
        c = rdtseg.rdt_seg_track(left_image, prob_image, idx,trans)
        if save == '1':
            cv.imwrite(os.path.join(dpath, '{0:06}.png'.format(idx)), c)
        slam.process_image_stereo(left_image[:, :, ::-1], right_image[:, :, ::-1], left_mask, right_mask, timestamp)
    i = 0
    result_path = 'rdr/d{}{}.txt'.format(sequence,i)
    while True:
        if not os.path.exists(result_path):
            s_flag = save_trajectory(slam.get_trajectory_points(), result_path)
            if s_flag:
                print(result_path)
                break
        i += 1
        result_path = 'rdr/d{}{}.txt'.format(sequence, i)

    slam.shutdown()
    mean_time = (time.time() - start_time) / num_images
    print('sequence ',sequence)
    print('mean process time: {}'.format(round(mean_time,2)))

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print('Usage: ./orbslam_stereo_kitti path_to_orb path_to_data device save_img sequence ')
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])