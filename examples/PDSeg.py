import cv2 as cv
import numpy as np


class PDSeg():
    def __init__(self, iml, coco_demo, disp_path, kernel):
        self.coco_demo = coco_demo
        self.disp_path = disp_path
        self.h, self.w = iml.shape[:2]

    def pd_seg(self,iml,prob_map):
        er = prob_map[..., 0].copy()
        er[er < 128] = 0
        er[er >= 128] = 255

        h, w = iml.shape[:2]
        a = self.coco_demo.compute_prediction(iml)
        top = self.coco_demo.select_top_predictions(a)
        masks = top.get_field("mask").numpy()
        labels = top.get_field("labels").numpy()

        c = np.zeros((h, w), dtype=np.uint8)
        for i in range(len(masks)):
            if labels[i] in {1, 2, 3, 4, 6, 8}:
                mask = masks[i].squeeze()
                box = top.bbox[i]
                x1, y1, x2, y2 = map(int, box)
                if 2 * (y2 - y1) > x2 - x1:
                    mi, ma = self.get_max_min_idx(er, w, y2)
                    xy1, xy2 = x1, x2
                    hw = w // 2
                else:
                    mi, ma = self.get_max_min_idx(er, h, x2)
                    xy1, xy2 = y1, y2
                    hw = h // 2
                if (mi != hw or ma != hw):
                    if labels[i] in {1, 2}:
                        if abs(xy2 - mi) <= (xy2 - xy1) or abs(xy1 - ma) <= (xy2 - xy1):
                            c[mask] = 255
                    if xy1 >= mi and xy2 <= ma:
                        c[mask] = 255
        return c

    def get_max_min_idx(self, er, cr, xy):
        fl, fr = 0, 0
        l, r = 0, cr - 1
        f = (cr == self.w)
        while True:
            if l < cr:
                if f:
                    if er[xy, l] == 0:
                        l += 1
                    else:
                        fl = 1
                else:
                    if er[l, xy] == 0:
                        l += 1
                    else:
                        fl = 1
            if r >= 0:
                if f:
                    if er[xy, r] == 0:
                        r -= 1
                    else:
                        fr = 1
                else:
                    if er[r, xy] == 0:
                        r -= 1
                    else:
                        fr = 1
            if l >= r or (fl and fr):
                break
        return l, r