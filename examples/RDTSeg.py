import cv2 as cv
import numpy as np
from copy import deepcopy as dp
import os


class RTSeg():
    def __init__(self, iml, coco, depth_path, kernel,config):
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
        self.fb_params = dict(pyr_scale=0.5,
                         levels=3,
                         winsize=3,
                         iterations=3,
                         poly_n=5,
                         poly_sigma=1.2,
                         flags=0)

        self.p_color = (0, 0, 255)

        self.mtx, self.dist = np.array([[707.0912, 0, 601.8873], [0, 707.0912, 183.1104], [0, 0, 1]]), np.array([[0] * 4]).reshape(1, 4).astype(np.float32)
        self.cverrs = []
        self.feature_params = dict(maxCorners=1000,
                          qualityLevel=0.1,
                          minDistance=7,
                          blockSize=7)
        self.paraml = {'minDisparity': 1,
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
        self.config = config


    def rt_seg_t(self,iml,prob_map):
        er = prob_map[..., 0].copy()
        er[er < 244] = 0
        er[er >= 244] = 255

        nr = prob_map.copy()
        nr[prob_map[..., 0] >= 244] = [0, 255, 0]
        nr[prob_map[..., 0] < 244] = [0, 0, 0]

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
                x1, y1, x2, y2 = top.bbox[i]
                res = self.get_max_min_idx(er, x1, y1, x2, y2)
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

    def get_max_min_idx(self, er, x1, y1, x2, y2):
        x1, y1, x2, y2 = map(int,[x1, y1, x2, y2])
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
        return res

    def limit(self,xy,f):
        if f:
            return max(min(xy,self.h-1),0)
        else:
            return max(min(xy, self.w - 1), 0)

    def rt_seg_track(self,iml,prob_map,idx):
        er = prob_map[..., 0].copy()
        er[er < 244] = 0
        er[er >= 244] = 255

        frame_gray = cv.cvtColor(iml, cv.COLOR_BGR2GRAY)
        nobj = len(self.obj)
        res = [True] * nobj
        for i in range(nobj):
            cm = np.where(self.obj[i][0] == True)
            if len(cm):
                y1, x1, y2, x2,  = self.obj[i][5]
                # print(y1, x1, y2, x2)
                flow = cv.calcOpticalFlowFarneback(self.old_gray, frame_gray, None, **self.fb_params)
                nm = np.zeros_like(self.obj[i][0], dtype=np.bool)
                dy, dx = np.mean(flow[self.obj[i][0]], axis=0)
                self.obj[i][5] = [self.limit(y1+dy,0),self.limit(x1+dx,1),self.limit(y2+dy,0),self.limit(x2+dx,1)]
                for x, y in zip(cm[0], cm[1]):
                    cx, cy = self.limit(x+dx,1), self.limit(y+dy,0)
                    nm[round(cx),round(cy)] = True
                self.obj[i][0] = nm
            else:
                res[i] = False

        self.obj = list(self.obj[res])
        self.track_obj(iml, idx)

        nobj = len(self.obj)

        for i in range(nobj):
            x1, y1, x2, y2 = self.obj[i][5]
            res= self.get_max_min_idx(er, x1, y1, x2, y2)
            if x1 >= 90 and x2 <= self.w - 90 and y2 >= 213:
                self.obj[i][1] += 1
                for mi, ma in res:
                    if self.obj[i][4] in self.sides_moving_labels:
                        if mi <= x1 <= ma or mi <= x2 <= ma:
                            self.obj[i][2] += 1
                    elif x1 >= mi and x2 <= ma:
                        self.obj[i][2] += 1
        c = np.ones((self.h, self.w),dtype=np.uint8)
        res = [True] * nobj

        for i in range(nobj):
            if self.obj[i][4] in self.cars:
                box = self.obj[i][5]
                x1, y1, x2, y2 = map(int, box)
                if idx - self.obj[i][3] >= 5 or (idx - self.obj[i][3] and (np.sum(self.obj[i][0]) < self.obj[i][6] or x1 <= 15 or x2 >= self.w - 15 or y1 <= 15 or y2 >= self.h - 15)):
                    res[i] = False
                elif self.obj[i][1] and self.obj[i][2] / self.obj[i][1] >= 0.6:
                    c[self.obj[i][0]] = 0
            elif idx - self.obj[i][3]:
                res[i] = False
            elif self.obj[i][1] and self.obj[i][2] / self.obj[i][1] >= 0.6:
                c[self.obj[i][0]] = 0

            # else:
            #     self.obj[i][2] = max(0, self.obj[i][2] - 0.5)
        self.obj = np.array(self.obj, dtype=object)
        self.obj = self.obj[res]
        nobj = len(self.obj)
        print('num of objs', nobj)
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
                    self.obj[x[1]][6] = np.sum(self.obj[x[1]][0])
                    self.obj[x[1]][0] = masks[x[2]][0] #.astype(np.bool)
                    self.obj[x[1]][3] = idx
                    self.obj[x[1]][5] = x[4]
                    nu_obj[x[1]] = False
                    nu_mask[x[2]] = False
                else:
                    break
        for i in range(nm):
            if nu_mask[i]:
                self.obj.append([masks[i][0], 0, 0, idx, masks[i][1],masks[i][2], 0]) # mask, appear, dyn, idx, label, box, last_mask
        # self.track_rate(idx)
        return

class RDTSeg(RTSeg):
    def __init__(self,iml, coco, depth_path, kernel,config):
        super(RDTSeg, self).__init__(iml, coco, depth_path, kernel,config)
        self.Q = self.getRectifyTransform()

    def stereoMatchSGBM(self, iml, imr):
        left_matcher = cv.StereoSGBM_create(**self.paraml)

        disparity_left = left_matcher.compute(iml, imr)

        trueDisp_left = disparity_left.astype(np.float32) / 16.

        return trueDisp_left

    def get_points(self, i, iml, imr):
        iml_, imr_ = preprocess(iml, imr)
        disp = self.stereoMatchSGBM(iml_, imr_)
        dis = cv.imread(os.path.join(self.depth_path, str(i).zfill(6) + '.png'))[..., 0]
        disp[disp == 0] = dis[disp == 0]
        points = cv.reprojectImageTo3D(disp, self.Q)
        return points

    def update(self, iml, imr, i, trans):
        self.old_gray = cv.cvtColor(iml, cv.COLOR_BGR2GRAY)
        self.p = cv.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
        self.p1 = dp(self.p)
        self.ast = np.ones((self.p.shape[0], 1))
        self.points = self.get_points(i, iml, imr)
        self.otfm = np.linalg.inv(trans)

    def projection(self, trans, frame_gray):
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p1, None, **self.lk_params)
        self.ast *= st
        tfm = trans
        tfm = self.otfm.dot(tfm)
        b = cv.Rodrigues(tfm[:3, :3])
        R = b[0]
        t = tfm[:3, 3].reshape((3, 1))

        P = p1[self.ast == 1]
        objpa = np.array([self.points[int(y), int(x)] for x, y in self.p[self.ast == 1].squeeze()])
        imgpts, jac = cv.projectPoints(objpa, R, t, self.mtx, self.dist)
        imgpts = imgpts.squeeze()
        P = P.squeeze()[~np.isnan(imgpts).any(axis=1)]
        imgpts = imgpts[~np.isnan(imgpts).any(axis=1)]
        P = P[(0 < imgpts[:, 0]) * (imgpts[:, 0] < self.w) * (0 < imgpts[:, 1]) * (imgpts[:, 1] < self.h)]
        imgpts = imgpts[(0 < imgpts[:, 0]) * (imgpts[:, 0] < self.w) * (0 < imgpts[:, 1]) * (imgpts[:, 1] < self.h)]
        error = ((P - imgpts) ** 2).sum(-1)
        P = P[error < 1e6]
        imgpts = imgpts[error < 1e6].astype(np.float32)
        error = error[error < 1e6]

        if len(imgpts):
            cverror = cv.norm(P, imgpts, cv.NORM_L2) / len(imgpts)
        else:
            cverror = float('inf')
        print(cverror)
        self.cverrs.append(cverror)
        self.p1 = p1
        ge = norm(error,imgpts)
        return ge, P

    def getRectifyTransform(self):
        left_K = self.config.cam_matrix_left
        right_K = self.config.cam_matrix_right
        left_distortion = self.config.distortion_l
        right_distortion = self.config.distortion_r
        R = self.config.R
        T = self.config.T

        R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                         (self.w, self.h), R, T, alpha=0)
        return Q

    def rdt_seg_track(self,iml,prob_map,idx,trans):
        er = prob_map[..., 0].copy()
        er[er < 244] = 0
        er[er >= 244] = 255

        frame_gray = cv.cvtColor(iml, cv.COLOR_BGR2GRAY)
        nobj = len(self.obj)
        res = [True] * nobj
        for i in range(nobj):
            cm = np.where(self.obj[i][0] == True)
            if len(cm):
                y1, x1, y2, x2,  = self.obj[i][5]
                # print(y1, x1, y2, x2)
                flow = cv.calcOpticalFlowFarneback(self.old_gray, frame_gray, None, **self.fb_params)
                nm = np.zeros_like(self.obj[i][0], dtype=np.bool)
                dy, dx = np.mean(flow[self.obj[i][0]], axis=0)
                self.obj[i][5] = [self.limit(y1+dy,0),self.limit(x1+dx,1),self.limit(y2+dy,0),self.limit(x2+dx,1)]
                for x, y in zip(cm[0], cm[1]):
                    cx, cy = self.limit(x+dx,1), self.limit(y+dy,0)
                    nm[round(cx),round(cy)] = True
                # print(self.obj[i][5])
                self.obj[i][0] = nm
            else:
                res[i] = False

        self.obj = list(self.obj[res])
        self.track_obj(iml, idx)

        ge, P = self.projection(trans, frame_gray)

        nobj = len(self.obj)

        for i in range(nobj):
            x1, y1, x2, y2 = self.obj[i][5]
            if x1 >= 90 and x2 <= self.w - 90:
                if y2 >= 213:
                    res = self.get_max_min_idx(er, x1, y1, x2, y2)
                    self.obj[i][1] += 1
                    for mi, ma in res:
                        if self.obj[i][4] in self.sides_moving_labels:
                            if abs(x2 - mi) <= (x2 - x1) or abs(x1 - ma) <= (x2 - x1) or (x1 >= mi and x2 <= ma):
                                self.obj[i][2] += 1
                        elif x1 >= mi and x2 <= ma:
                            self.obj[i][2] += 1
                else:
                    ao = 0
                    co = 0
                    for gi in range(len(ge)):
                        x, y = round(P[gi][1]), round(P[gi][0])
                        if 0 <= x < self.h and 0 <= y < self.w and self.obj[i][0][x, y]:
                            ao += 1
                            if ge[gi]:
                                co += 1
                    if ao > 1 and co / ao > 0.5:
                        self.obj[i][2] += 1
                        self.obj[i][1] += 1

        c = np.ones((self.h, self.w),dtype=np.uint8)
        res = [True] * nobj

        for i in range(nobj):
            if self.obj[i][4] in self.cars:
                box = self.obj[i][5]
                x1, y1, x2, y2 = map(int, box)
                if idx - self.obj[i][3] >= 5 or (idx - self.obj[i][3] and (np.sum(self.obj[i][0]) < self.obj[i][6] or x1 <= 15 or x2 >= self.w - 15 or y1 <= 15 or y2 >= self.h - 15)):
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
        nobj = len(self.obj)
        print('num of objs', nobj)
        for obj in self.obj:
            print('a: {}, d: {}'.format(obj[1], obj[2]))
        self.old_gray = frame_gray.copy()
        return c



def Rt_to_tran(tfm):
    res = np.zeros((4, 4))
    res[:3, :] = tfm[:3, :]
    res[3, 3] = 1
    return res


def preprocess(img1, img2):
    im1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    im2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    im1 = cv.equalizeHist(im1)
    im2 = cv.equalizeHist(im2)

    return im1, im2


def get_IOU(m1, m2):
    I = np.sum(np.logical_and(m1, m2))
    U = np.sum(np.logical_or(m1, m2))
    s1 = np.sum(m1)
    s2 = np.sum(m2)
    if s1 and s2:
        s = s1 / s2 if s1 > s2 else s2 / s1
        if s > 2.7:
            return 0
        U *= s
    else:
        U = 0
    if U:
        return I / U
    else:
        return 0

def norm(error, imgpts):
    merror = np.array(error)
    lma = imgpts[:, 0] < 500

    rma = imgpts[:, 0] > 740

    mma = np.logical_and((~lma), (~rma))

    ge = np.array([False] * len(merror))
    lm = merror[lma]
    rm = merror[rma]
    mm = merror[mma]
    if len(lm):
        ge[lma] = lm > np.percentile(lm, 93)
    if len(rm):
        ge[rma] = rm > np.percentile(rm, 93)
    if len(mm):
        ge[mma] = mm > np.percentile(mm, 81)
    return ge