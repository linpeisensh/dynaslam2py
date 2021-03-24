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
        self.old_gray = cv.cvtColor(iml, cv.COLOR_BGR2GRAY)

        self.kernel = kernel
        self.lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        self.p_color = (0, 0, 255)

    def pd_seg(self,iml,prob_map):
        er = prob_map[..., 0].copy()
        er[er < 128] = 0
        er[er >= 128] = 255

        h, w = iml.shape[:2]
        a = self.coco.compute_prediction(iml)
        top = self.coco.select_top_predictions(a)
        masks = top.get_field("mask").numpy()
        labels = top.get_field("labels").numpy()

        c = np.ones((self.h, self.w),dtype=np.uint8)
        for i in range(len(masks)):
            if labels[i] in {1, 2, 3, 4, 6, 8}:
                mask = masks[i].squeeze()
                box = top.bbox[i]
                x1, y1, x2, y2 = map(int, box)
                if 2.25 * (y2 - y1) > x2 - x1:
                    mi, ma = self.get_max_min_idx(er, self.w, min(y2+10,self.h-1))
                    xy1, xy2 = x1, x2
                    hw = w // 2
                else:
                    if x2 < 500:
                        mi, ma = self.get_max_min_idx(er, self.h, min(x2 + 10, self.w - 1))
                    else:
                        mi, ma = self.get_max_min_idx(er, self.h, min(x1 - 10,self.w-1))
                    xy1, xy2 = y1, y2
                    hw = h // 2
                if (mi != hw or ma != hw):
                    if labels[i] in {1, 2}:
                        if abs(xy2 - mi) <= (xy2 - xy1) or abs(xy1 - ma) <= (xy2 - xy1):
                            c[mask] = 0
                    if xy1 >= mi and xy2 <= ma:
                        c[mask] = 0
        return c

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
            if labels[i] in {1, 2, 3, 4, 6, 8}:
                mask = masks[i].squeeze()
                box = top.bbox[i]
                x1, y1, x2, y2 = map(int, box)
                x = (x1 + x2) / 2
                if 400 < x < 836:
                    res = self.get_max_min_idx(er, self.w, min(y2 + 10, self.h - 1))
                    xy1, xy2 = x1, x2
                    cc = cv.circle(cc, (x1, y2), 5, self.p_color, -1)
                    cc = cv.circle(cc, (x2, y2), 5, self.p_color, -1)
                else:
                    if 2.25 * (y2 - y1) > x2 - x1:
                        res = self.get_max_min_idx(er, self.w, min(y2 + 10, self.h - 1))
                        xy1, xy2 = x1, x2
                        cc = cv.circle(cc, (x1, y2), 5, self.p_color, -1)
                        cc = cv.circle(cc, (x2, y2), 5, self.p_color, -1)
                    else:
                        res = self.get_max_min_idx(er, self.h, min(x2 + 10, self.w - 1))
                        xy1, xy2 = y1, y2
                        cc = cv.circle(cc, (x2, y1), 5, self.p_color, -1)
                        cc = cv.circle(cc, (x2, y2), 5, self.p_color, -1)
                print(y2,res)
                for mi, ma in res:
                    if labels[i] in {1, 2}:
                        if abs(xy2 - mi) <= (xy2 - xy1) or abs(xy1 - ma) <= (xy2 - xy1) or (
                                xy1 >= mi and xy2 <= ma):
                            cc[mask, ...] = 255
                    elif xy1 >= mi and xy2 <= ma:
                        cc[mask, ...] = 255
        return cc

    def get_max_min_idx(self, er, cr, xy):
        res = []
        l = 0
        r = 1
        f = (cr == self.w)
        while r < cr:
            if f:
                while l < cr and er[xy, l] == 0:
                    l += 1
                lr = r
                r =  l + 1
                while r < cr and er[xy, r] == 255:
                    r += 1
                if r < cr and er[xy, r] == 255 and res and l - lr < cr / 4:
                    l, _ = res.pop()
            else:
                while l < cr and er[l,xy] == 0:
                    l += 1
                lr = r
                r = l + 1
                while r < cr and er[r,xy] == 255:
                    r += 1
                if r < cr and er[xy, r] == 255 and res and l - lr < cr / 4:
                    l, _ = res.pop()
            if r - l > 2:
                res.append([l, r - 1])
            l = r
        return res

    # def pd_seg_t(self, iml, prob_map):
    #     er = prob_map[..., 0].copy()
    #     er[er < 244] = 0
    #     er[er >= 244] = 255
    #
    #     nr = prob_map.copy()
    #     nr[prob_map[..., 0] > 244] = [0, 255, 0]
    #     nr[prob_map[..., 0] <= 244] = [0, 0, 0]
    #
    #     a = self.coco.compute_prediction(iml)
    #     top = self.coco.select_top_predictions(a)
    #     masks = top.get_field("mask").numpy()
    #     labels = top.get_field("labels").numpy()
    #
    #     c = np.zeros((self.h, self.w), dtype=np.uint8)
    #     cc = np.repeat(c[:, :, np.newaxis], 3, axis=2)
    #     cc = cv.add(cc, nr)
    #     for i in range(len(masks)):
    #         if labels[i] in {1, 2, 3, 4, 6, 8}:
    #             mask = masks[i].squeeze()
    #             box = top.bbox[i]
    #             x1, y1, x2, y2 = map(int, box)
    #             # if x2 > 500:
    #             if 2.25 * (y2 - y1) > x2 - x1:
    #                 mi, ma = self.get_max_min_idx(er, self.w, min(y2 + 5, self.h - 1))
    #                 xy1, xy2 = x1, x2
    #                 cc = cv.circle(cc, (x1, y2), 5, self.p_color, -1)
    #                 cc = cv.circle(cc, (x2, y2), 5, self.p_color, -1)
    #                 hw = self.w // 2
    #             else:
    #                 if x2 < 500:
    #                     mi, ma = self.get_max_min_idx(er, self.h, min(x2 + 5, self.w - 1))
    #                     x = x2
    #                 else:
    #                     mi, ma = self.get_max_min_idx(er, self.h, min(x1 - 5, self.w - 1))
    #                     x = x1
    #                 xy1, xy2 = y1, y2
    #                 cc = cv.circle(cc, (x, y1), 5, self.p_color, -1)
    #                 cc = cv.circle(cc, (x, y2), 5, self.p_color, -1)
    #                 hw = self.h // 2
    #             # else:
    #             #     mi, ma = self.get_max_min_idx(er, self.w, y2)
    #             #     xy1, xy2 = x1, x2
    #             #     cc = cv.circle(cc, (x1, y2), 5, self.p_color, -1)
    #             #     cc = cv.circle(cc, (x2, y2), 5, self.p_color, -1)
    #             #     hw = self.w // 2
    #             print(mi,ma)
    #             if (mi != hw or ma != hw):
    #                 if labels[i] in {1, 2}:
    #                     if abs(xy2 - mi) <= (xy2 - xy1) or abs(xy1 - ma) <= (xy2 - xy1):
    #                         cc[mask, ...] = 255
    #                 if xy1 >= mi and xy2 <= ma:
    #                     cc[mask, ...] = 255
    #     return cc

    # def get_max_min_idx(self, er, cr, xy):
    #     fl, fr = 0, 0
    #     l, r = 0, cr - 1
    #     f = (cr == self.w)
    #     while True:
    #         if l < cr:
    #             if f:
    #                 if er[xy, l] == 0:
    #                     l += 1
    #                 else:
    #                     fl = 1
    #             else:
    #                 if er[l, xy] == 0:
    #                     l += 1
    #                 else:
    #                     fl = 1
    #         if r >= 0:
    #             if f:
    #                 if er[xy, r] == 0:
    #                     r -= 1
    #                 else:
    #                     fr = 1
    #             else:
    #                 if er[r, xy] == 0:
    #                     r -= 1
    #                 else:
    #                     fr = 1
    #         if l >= r or (fl and fr):
    #             break
    #     return l, r

    def pd_seg_rec(self,iml,prob_map,idx):
        er = prob_map[..., 0].copy()
        er[er < 128] = 0
        er[er >= 128] = 255

        frame_gray = cv.cvtColor(iml, cv.COLOR_BGR2GRAY)
        nobj = len(self.obj)
        for i in range(nobj):
            cm = np.where(self.obj[i][0] == True)
            cmps = np.array(list(zip(cm[1], cm[0]))).astype(np.float32)
            nmps, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, frame_gray, cmps, None, **self.lk_params)
            nm = np.zeros_like(self.obj[i][0], dtype=np.uint8)
            for nmp in nmps:
                x, y = round(nmp[1]), round(nmp[0])
                if 0 <= x < self.h and 0 <= y < self.w:
                    nm[x, y] = 1
            nm = cv.erode(cv.dilate(nm, self.kernel), self.kernel)
            self.obj[i][0] = nm.astype(np.bool)

        self.obj = list(self.obj)
        self.track_obj(iml, idx)

        nobj = len(self.obj)

        for i in range(nobj):
            box = self.obj[i][5]
            x1, y1, x2, y2 = map(int, box)
            x = (x1 + x2) / 2
            if 400 < x < 836:
                res = self.get_max_min_idx(er, self.w, min(y2 + 10, self.h - 1))
                xy1, xy2 = x1, x2
            else:
                if 2.25 * (y2 - y1) > x2 - x1:
                    res = self.get_max_min_idx(er, self.w, min(y2 + 10, self.h - 1))
                    xy1, xy2 = x1, x2
                else:
                    res = self.get_max_min_idx(er, self.h, min(x2 + 10, self.w - 1))
                    xy1, xy2 = y1, y2
            for mi, ma in res:
                if self.obj[i][4] in {1, 2}:
                    if abs(xy2 - mi) <= (xy2 - xy1) or abs(xy1 - ma) <= (xy2 - xy1) or (xy1 >= mi and xy2 <= ma):
                        self.obj[i][2] += 1
                elif xy1 >= mi and xy2 <= ma:
                    self.obj[i][2] += 1
        c = np.ones((self.h, self.w),dtype=np.uint8)
        res = [True] * nobj
        print('num of objs', nobj)
        for i in range(nobj):
            if idx - self.obj[i][3] != 0:
                res[i] = False
            elif self.obj[i][2] / self.obj[i][1] >= 0.6 or self.obj[i][2] >= 5:  #
                c[self.obj[i][0]] = 0
            else:
                self.obj[i][2] = max(0, self.obj[i][2] - 0.5)
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
                mask = mask.astype(np.uint8)
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
                    self.obj[x[1]][0] = masks[x[2]][0].astype(np.bool)
                    self.obj[x[1]][1] += 1
                    self.obj[x[1]][3] = idx
                    self.obj[x[1]][5] = x[4]
                    nu_obj[x[1]] = False
                    nu_mask[x[2]] = False
                else:
                    break
        for i in range(nm):
            if nu_mask[i]:
                self.obj.append([masks[i][0].astype(np.bool), 1, 0, idx, masks[i][1],masks[i][2]]) # mask, appear, dyn, idx, label, box
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