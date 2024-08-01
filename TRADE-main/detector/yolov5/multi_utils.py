import numpy as np
import torch
import torchvision.transforms as T
import os
from PIL import Image
from config import cfg
from yacs.config import CfgNode
from reid.reid_inference.reid_model import build_reid_model

def add_zone_num(lines,zones):
    mot_list = dict()
    for line in lines:
        fid = int(lines[line]['frame'][3:])
        tid = lines[line]['id']
        bbox = list(map(lambda x:int(float(x)), lines[line]['bbox']))
        if tid not in mot_list:
            mot_list[tid] = dict()
        out_dict = lines[line]
        out_dict['zone'] = zones.get_zone(bbox) # 给bbox分配了zone_num
        mot_list[tid][fid] = out_dict # 字典
    return mot_list # 字典

def gather_sequence_info(det_feat_dic): 
    bbox_dic = {}
    feat_dic = {}
    for image_name in det_feat_dic:
        frame_index = int(image_name.split('_')[1])
        det_bbox = np.array(det_feat_dic[image_name]['bbox']).astype('float32')
        det_feat = det_feat_dic[image_name]['feat']
        score = det_feat_dic[image_name]['conf']
        score = np.array((score,))
        det_bbox = np.concatenate((det_bbox, score)).astype('float32')
        if frame_index not in bbox_dic:
            bbox_dic[frame_index] = [det_bbox]
            feat_dic[frame_index] = [det_feat]
        else:
            bbox_dic[frame_index].append(det_bbox)
            feat_dic[frame_index].append(det_feat)
    seq_info = {
        "detections": [bbox_dic, feat_dic],
    }
    return seq_info

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    # clip_coords(coords, img0_shape)
    return coords

def cfg_extract():
    cfg = CfgNode()
    cfg.REID_MODEL= 'detector/yolov5/reid/reid_model/resnet101_ibn_a_2.pth'
    cfg.REID_BACKBONE= 'resnet101_ibn_a'
    cfg.REID_SIZE_TEST= [384, 384]
    cfg.freeze()
    return cfg

class ReidFeature():
    """Extract reid feature."""

    def __init__(self, gpu_id, _mcmt_cfg):
        print("init reid model")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.model, self.reid_cfg = build_reid_model(_mcmt_cfg)
        device = torch.device('cuda')
        print('device: ', device)
        self.model = self.model.to(device)
        self.model.eval()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.val_transforms = T.Compose([T.Resize(self.reid_cfg.INPUT.SIZE_TEST, interpolation=3),\
                              T.ToTensor(), T.Normalize(mean=mean, std=std)])

    def extract(self, img_dict):
        """Extract image feature with given image path.
        Feature shape (2048,) float32."""

        img_batch = []
        for _, img0 in img_dict.items():
            img = Image.fromarray(img0)
            img = self.val_transforms(img)
            img = img.unsqueeze(0)
            img_batch.append(img)
        img = torch.cat(img_batch, dim=0)

        with torch.no_grad():
            img = img.to('cuda')
            flip_feats = False
            if self.reid_cfg.TEST.FLIP_FEATS == 'yes': flip_feats = True
            if flip_feats:
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                        feat1 = self.model(img)
                    else:
                        feat2 = self.model(img)
                feat = feat2 + feat1
            else:
                feat = self.model(img)
        feat = feat.cpu().detach().numpy()
        return feat
