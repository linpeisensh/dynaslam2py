from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np



class PSMNet():
    def __init__(self,loadmodel,model):
        self.model = self.pre(loadmodel,model)
        self.processed = preprocess.get_transform(augment=False)

    def pre(self, loadmodel, model):
        if model == 'stackhourglass':
            model = stackhourglass(192)
        elif model == 'basic':
            model = basic(192)
        else:
            print('no model')

        model = nn.DataParallel(model, device_ids=[0])
        model.cuda()
        if loadmodel is not None:
            state_dict = torch.load(loadmodel)
            model.load_state_dict(state_dict['state_dict'])

        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
        return model

    def test(self, imgL, imgR):
        self.model.eval()

        imgL = torch.FloatTensor(imgL).cuda()
        imgR = torch.FloatTensor(imgR).cuda()

        imgL, imgR = Variable(imgL), Variable(imgR)

        with torch.no_grad():
            output = self.model(imgL, imgR)
        output = torch.squeeze(output)
        pred_disp = output.data.cpu().numpy()

        return pred_disp

    def main(self, test_left_img, test_right_img):
        imgL_o = test_left_img[...,[2,1,0]].astype('float32')
        imgR_o = test_right_img[...,[2,1,0]].astype('float32')
        imgL = self.processed(imgL_o).numpy()
        imgR = self.processed(imgR_o).numpy()
        imgL = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
        imgR = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])

        # pad to (384, 1248)
        top_pad = 384 - imgL.shape[2]
        left_pad = 1248 - imgL.shape[3]
        imgL = np.lib.pad(imgL, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)
        imgR = np.lib.pad(imgR, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)

        pred_disp = self.test(imgL, imgR)

        top_pad = 384 - imgL_o.shape[0]
        left_pad = 1248 - imgL_o.shape[1]
        img = pred_disp[top_pad:, :-left_pad]
        return img

if __name__ == '__main__':
    import cv2 as cv
    import os
    from psmnet.utils import preprocess
    from psmnet.models import *

    torch.manual_seed(1)
    lm = './finetune_300.tar'
    model = 'stackhourglass'
    psmnet = PSMNet(lm,model)
    image_path = '/storage/remote/atcremers17/linp/dataset/kittic/sequences/10/'
    iml = cv.imread(os.path.join(image_path,'image_2','000000.png'), cv.IMREAD_UNCHANGED)
    imr = cv.imread(os.path.join(image_path,'image_3','000000.png'), cv.IMREAD_UNCHANGED)
    res = psmnet.main(iml,imr)
    print(res.shape)








