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

import cv2 as cv
import traceback
import os
import shutil

def load_images(path_to_sequence):
    res = [os.path.join(path_to_sequence, img) for img in os.listdir(path_to_sequence)]
    res.sort()
    return res

sequence = '04'

file_path = os.path.join('/storage/remote/atcremers17/linp/dataset/kittic/sequences/',sequence, 'image_2')
left_filenames = load_images(file_path)

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
pdseg = PDSeg(iml,coco_demo,depth_path,kernel)

num_images = len(left_filenames)

dpath = 'pmask/t{}/'.format(sequence)
if os.path.exists(dpath):
    shutil.rmtree(dpath)
os.mkdir(dpath)

for idx in range(num_images):
    left_image = cv.imread(left_filenames[idx], cv.IMREAD_UNCHANGED)
    prob_image = cv.imread(prob_filenames[idx])
    try:
        c = pdseg.pd_seg_t(left_image, prob_image)
        cv.imwrite(os.path.join(dpath, '{0:06}.png'.format(idx)), c)
    except:
        traceback.print_exc()
    print('{} frame'.format(idx))
