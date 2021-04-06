import cv2 as cv
import numpy as np
import sys
import os

from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo

config_file = '../../maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml'
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", 'cuda'])
coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

fb_params = dict(pyr_scale = 0.5,
                 levels = 3,
                 winsize = 3,
                 iterations = 3,
                 poly_n = 5,
                 poly_sigma = 1.2,
                 flags = 0)
sequence = sys.argv[1]
s = int(sys.argv[2])

iml = cv.imread('/storage/remote/atcremers17/linp/dataset/kittic/sequences/{}/'.format(sequence)+'image_2/{0:06}.png'.format(s))
a = coco_demo.compute_prediction(iml)
top = coco_demo.select_top_predictions(a)
masks = top.get_field("mask").numpy()
labels = top.get_field("labels").numpy()
h,w = iml.shape[:2]
p_color = (0, 0, 255) # BGR
o_color = (0, 255, 0)
res = []
biml = iml.copy()
for i in range(len(masks)):
    if labels[i] in {1,2,3,4,6,8}:
        box = top.bbox[i]
        x1,y1,x2,y2 = map(int,box)
        mask = masks[i].squeeze()
        res.append([mask,x1,y1,x2,y2])
        biml[mask] = o_color
cv.imwrite('dol/{0:06}.png'.format(s),biml)

old_gray = cv.cvtColor(iml, cv.COLOR_BGR2GRAY)

n = 5
for i in range(1,n):
    ci = s+i
    print('{} frame'.format(ci))
    cr = []
    iml = cv.imread('/storage/remote/atcremers17/linp/dataset/kittic/sequences/{}/'.format(sequence)+'image_2/{0:06}.png'.format(ci))
    frame_gray = cv.cvtColor(iml, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(old_gray, frame_gray, None, **fb_params)
    # res, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, res, None, **lk_params)
    for mask,  y1, x1, y2, x2 in res:
        dy, dx = np.mean(flow[mask],axis=0)
        cmp = np.where(mask==True)
        nmask = np.zeros_like(mask,dtype=np.bool)
        for x, y in zip(cmp[0],cmp[1]):
            cx = max(min(round(x+dx),h-1),0)
            cy = max(min(round(y+dy),w-1),0)
            nmask[cx,cy] = True
        cr.append([nmask,max(min(round(y1+dy),w-1),0),max(min(round(x1+dx),h-1),0),max(min(round(y2+dy),w-1),0),max(min(round(x2+dx),h-1),0)])
    res = cr
    old_gray = frame_gray
    a = coco_demo.compute_prediction(iml)
    top = coco_demo.select_top_predictions(a)
    masks = top.get_field("mask").numpy()
    labels = top.get_field("labels").numpy()
    biml = iml.copy()
    for i in range(len(masks)):
        if labels[i] in {1,2,3,4,6,8}:
            box = top.bbox[i]
            x1,y1,x2,y2 = map(int,box)
            biml = cv.circle(biml,(x1,y1),5,o_color,-1)
            biml = cv.circle(biml,(x2,y2),5,o_color,-1)
    for mask, x1, y1, x2, y2 in res:
        biml = cv.circle(biml,(x1,y1),5,p_color,-1)
        biml = cv.circle(biml,(x2,y2),5,p_color,-1)
        biml[mask] = p_color
    cv.imwrite('dol/{0:06}.png'.format(ci),biml)
