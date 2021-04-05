import cv2 as cv
import numpy as np


class PDSeg():
    def __init__(self, iml, coco, depth_path, kernel):
        self.coco = coco
        self.depth_path = depth_path
        self.h, self.w = iml.shape[:2]

        self.obj = np.array([])
        self.sides_moving_labels = {1,2}
        self.pot_moving_labels = {1,2,3,4,6,8}
        self.cars = {3,6,8}
        self.old_gray = cv.cvtColor(iml, cv.COLOR_BGR2GRAY)

        self.kernel = kernel
        self.lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        self.p_color = (0, 0, 255)
        self.er = np.zeros((self.h,self.w))

    # def pd_seg(self,iml,prob_map):
    #     er = prob_map[..., 0].copy()
    #     er[er < 128] = 0
    #     er[er >= 128] = 255
    #
    #     a = self.coco.compute_prediction(iml)
    #     top = self.coco.select_top_predictions(a)
    #     masks = top.get_field("mask").numpy()
    #     labels = top.get_field("labels").numpy()
    #
    #     c = np.ones((self.h, self.w),dtype=np.uint8)
    #     for i in range(len(masks)):
    #         if labels[i] in self.pot_moving_labels:
    #             mask = masks[i].squeeze()
    #             box = top.bbox[i]
    #             x1, y1, x2, y2 = map(int, box)
    #             x = (x1 + x2) / 2
    #             if 400 < x < 836:
    #                 res = self.get_max_min_idx(er, self.w, min(y2 + 10, self.h - 1))
    #                 x1, x2 = x1, x2
    #             else:
    #                 if 2.25 * (y2 - y1) > x2 - x1:
    #                     res = self.get_max_min_idx(er, self.w, min(y2 + 10, self.h - 1))
    #                     x1, x2 = x1, x2
    #                 else:
    #                     res = self.get_max_min_idx(er, self.h, min(x2 + 10, self.w - 1))
    #                     x1, x2 = y1, y2
    #             for mi, ma in res:
    #                 if labels[i] in self.sides_moving_labels:
    #                     if abs(x2 - mi) <= (x2 - x1) or abs(x1 - ma) <= (x2 - x1) or (
    #                             x1 >= mi and x2 <= ma):
    #                         c[mask] = 0
    #                 elif x1 >= mi and x2 <= ma:
    #                     c[mask] = 0
    #     return c

    def pd_seg_t(self,iml,prob_map):
        er = prob_map[..., 0].copy()
        er[er < 244] = 0
        er[er >= 244] = 255

        nr = prob_map.copy()
        nr[prob_map[..., 0] > 244] = [0, 255, 0]
        nr[prob_map[..., 0] <= 244] = [0, 0, 0]

        a = self.coco.compute_prediction(iml)
        top = self.coco.select_top_predictions(a)
        masks = top.get_field("mask").numpy()
        labels = top.get_field("labels").numpy()

        c = np.zeros((self.h, self.w), dtype=np.uint8)
        cc = np.repeat(c[:, :, np.newaxis], 3, axis=2)
        cc = cv.add(cc, nr)
        for i in range(len(masks)):
            if labels[i] in self.pot_moving_labels:
                mask = masks[i].squeeze()
                box = top.bbox[i]
                res, x1, x2, y1, y2 = self.get_max_min_idx(er, box)
                cc = cv.circle(cc, (x1, y2), 5, self.p_color, -1)
                cc = cv.circle(cc, (x2, y2), 5, self.p_color, -1)
                print(y2,res)
                for mi, ma in res:
                    if labels[i] in self.sides_moving_labels:
                        if abs(x2 - mi) <= (x2 - x1) or abs(x1 - ma) <= (x2 - x1) or (
                                x1 >= mi and x2 <= ma):
                            cc[mask, ...] = 255
                    elif (x1 >= mi and x2 <= ma and self.w - 90 >= x2 and x1 >= 90):
                        cc[mask, ...] = 255
        return cc

    def get_max_min_idx(self, er, box):
        x1, y1, x2, y2 = map(int, box)
        def helper(y2,f):
            res = []
            l = 0
            r = 1
            if self.h - y2 > 15:
                if f:
                    y = y2 + 10
                else:
                    y = y2 + 3
            else:
                y = y2
            while r < self.w:
                while l < self.w and er[y, l] == 0:
                    l += 1
                lr = r
                r =  l + 1
                while r < self.w and er[y, r] == 255:
                    r += 1
                if r <  self.w and er[y, r-1] == 255 and res and l - lr < max(1.05 * (x2 - x1),self.w/4):
                    l, _ = res.pop()
                if r - l > 2:
                    res.append([l, r - 1])
                l = r
            return res
        ans = helper(y2,0)
        if ans:
            x = (x1 + x2) / 2
            dis = float('inf')
            nm = -1
            for l, r in ans:
                m = (l + r) / 2
                cd = abs(m - x)
                if cd < dis:
                    dis = cd
                    nm = m
            if x1 < nm:
                res = helper(y2,1)
            else:
                res = helper(y2,0)
        else:
            res = ans
        return res, x1, x2, y1, y2

    def limit(self,xy,f):
        if f:
            return max(min(xy,self.h-1),0)
        else:
            return max(min(xy, self.w - 1), 0)

    def pd_seg_rec(self,iml,prob_map,idx):
        er = prob_map[..., 0].copy()
        er[er < 244] = 0
        er[er >= 244] = 255

        frame_gray = cv.cvtColor(iml, cv.COLOR_BGR2GRAY)
        nobj = len(self.obj)
        res = [True] * nobj
        for i in range(nobj):
            cm = np.where(self.obj[i][0] == True)
            cmps = np.array(list(zip(cm[1], cm[0]))).astype(np.float32)
            if len(cmps):
                y1, x1, y2, x2,  = self.obj[i][5]
                # print(y1, x1, y2, x2)
                nmps, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, frame_gray, cmps, None, **self.lk_params)
                nm = np.zeros_like(self.obj[i][0], dtype=np.uint8)
                idx = 0
                dx, dy = 0, 0
                for j,nmp in enumerate(nmps):
                    x, y = round(nmp[1]), round(nmp[0])
                    if 0 <= x < self.h and 0 <= y < self.w:
                        nm[x, y] = 1
                        idx += 1
                        dx += (x-cmps[i,1])
                        dy += (y-cmps[i,0])
                dx /= idx
                dy /= idx
                self.obj[i][5] = [self.limit(y1+dy,0),self.limit(x1+dx,1),self.limit(y2+dy,0),self.limit(x2+dx,1)]
                # print(self.obj[i][5])
                nm = cv.erode(cv.dilate(nm, self.kernel), self.kernel)
                self.obj[i][0] = nm.astype(np.bool)
            else:
                res[i] = False

        self.obj = list(self.obj[res])
        self.track_obj(iml, idx)

        nobj = len(self.obj)

        for i in range(nobj):
            if self.obj[i][6]:
                box = self.obj[i][5]
                res, x1, x2, y1, y2 = self.get_max_min_idx(er, box)
                for mi, ma in res:
                    if self.obj[i][4] in self.sides_moving_labels:
                        if abs(x2 - mi) <= (x2 - x1) or abs(x1 - ma) <= (x2 - x1) or (x1 >= mi and x2 <= ma):
                            self.obj[i][2] += 1
                    elif x1 >= mi and x2 <= ma:
                        self.obj[i][2] += 1
        c = np.ones((self.h, self.w),dtype=np.uint8)
        res = [True] * nobj
        print('num of objs', nobj)
        for i in range(nobj):
            if self.obj[i][4] in self.cars:
                box = self.obj[i][5]
                x1, y1, x2, y2 = map(int, box)
                if idx - self.obj[i][3] >= 5 or (idx - self.obj[i][3] and (np.sum(self.obj[i][0]) < self.obj[i][7] or x1 <= 45 or x2 >= self.w-45 or y1 <= 45 or y2 >= self.h - 45)):
                    res[i] = False
                elif self.obj[i][1] and self.obj[i][2] / self.obj[i][1] >= 0.6:  #  or self.obj[i][2] >= 5
                    c[self.obj[i][0]] = 0
            elif idx - self.obj[i][3]:
                res[i] = False
            elif self.obj[i][1] and self.obj[i][2] / self.obj[i][1] >= 0.6:  #  or self.obj[i][2] >= 5
                c[self.obj[i][0]] = 0

            # else:
            #     self.obj[i][2] = max(0, self.obj[i][2] - 0.5)
        self.obj = np.array(self.obj, dtype=object)
        self.obj = self.obj[res]
        for obj in self.obj:
            print('a: {}, d: {}'.format(obj[1], obj[2]))
        self.old_gray = frame_gray.copy()
        return c

    def track_obj(self, iml, idx):
        image = iml.astype(np.uint8)
        prediction = self.coco.compute_prediction(image)
        top = self.coco.select_top_predictions(prediction)
        omasks = top.get_field("mask").numpy()
        labels = list(map(int, top.get_field("labels")))
        nl = len(labels)
        masks = []
        self.omasks = np.ones((self.h, self.w), dtype=np.uint8)
        for i in range(nl):
            mask = omasks[i].squeeze()
            self.omasks[mask] = 0
            if labels[i] in self.pot_moving_labels:
                # mask = mask.astype(np.uint8)
                masks.append([mask,labels[i],top.bbox[i]])
        res = []
        nc = len(self.obj)
        nm = len(masks)
        for i in range(nm):
            for j in range(nc):
                cIOU = get_IOU(masks[i][0], self.obj[j][0])
                res.append((cIOU, j, i, masks[i][1],masks[i][2]))
        nu_obj = [True] * nc
        nu_mask = [True] * nm
        res.sort(key=lambda x: -x[0])
        for x in res:
            if nu_obj[x[1]] and nu_mask[x[2]]:
                if x[0] > 0 and x[3] == self.obj[x[1]][4]:
                    self.obj[x[1]][7] = np.sum(self.obj[x[1]][0])
                    self.obj[x[1]][0] = masks[x[2]][0] #.astype(np.bool)
                    if x[4][0] >= 90 and x[4][2] <= self.w - 90 and x[4][3] >= 213:
                        self.obj[x[1]][1] += 1
                        self.obj[x[1]][6] = True
                    else:
                        self.obj[x[1]][6] = False
                    self.obj[x[1]][3] = idx
                    self.obj[x[1]][5] = x[4]
                    nu_obj[x[1]] = False
                    nu_mask[x[2]] = False
                else:
                    break
        for i in range(nm):
            if nu_mask[i]:
                if masks[i][2][0] >= 90 and masks[i][2][2] <= self.w - 90 and masks[i][2][3] >= 213: # .astype(np.bool)
                    self.obj.append([masks[i][0], 1, 0, idx, masks[i][1],masks[i][2], True,0]) # mask, appear, dyn, idx, label, box, in region, last_mask
                else: # .astype(np.bool)
                    self.obj.append([masks[i][0], 0, 0, idx, masks[i][1], masks[i][2], False,0])
        # self.track_rate(idx)
        return




    

def get_IOU(m1, m2):
    I = np.sum(np.logical_and(m1, m2))
    U = np.sum(np.logical_or(m1, m2))
    s1 = np.sum(m1)
    s2 = np.sum(m2)
    if s1 and s2:
        s = s1 / s2 if s1 > s2 else s2 / s1
        U *= s
    else:
        U = 0
    if U:
        return I / U
    else:
        return 0