import os
import sys
import queue
import logging
import traceback
from copy import deepcopy
import time
from pathlib import Path
import motmetrics as mm
import pickle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
from numpy import random
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from torchvision.ops import nms

from config import cfg
from yacs.config import CfgNode
from reid.matching.tools.utils.filter import *
from reid.matching.tools.utils.visual_rr import visual_rerank
from reid.matching.tools.utils.zone_intra import zone

from MOTBaseline.src.fm_tracker.multitracker import JDETracker
from MOTBaseline.src.post_processing.post_association import associate
from MOTBaseline.src.post_processing.track_nms import track_nms

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size,
    check_requirements,
    non_max_suppression,
    apply_classifier,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from reid.reid_inference.reid_model import build_reid_model
from utils.plots import plot_one_box
from utils.torch_utils import (
    select_device,
    load_classifier,
    time_synchronized,
    get_gpu_mem_info,
    get_cpu_mem_info,
)


# Namespace(agnostic_nms=True, augment=False, cfg_file='aic_all.yml', 
# classes=[2, 5, 7], conf_thres=0.1, device='', exist_ok=False, img_size=1280, 
# iou_thres=0.45, name='c041', project='/mnt/c/Users/83725/Desktop/AIC21-MTMC/datasets/detect_merge/', 
# save_conf=True, save_txt=True, source='/mnt/c/Users/83725/Desktop/AIC21-MTMC/datasets/detection/images/test/S06//c041/img1/', 
# update=False, view_img=False, weights=['yolov5s.pt'])

# formatted_date = time.strftime("%Y-%m-%d", time.localtime())
# log_name = f'town5/{formatted_date}_detect_res.log'
# logging.basicConfig(filename=log_name, level=logging.INFO, format='%(asctime)s - %(message)s')
GPU_ID = 3
conf_thres = 0.25
iou_thres = 0.45
vdo_frame_ratio = 10
min_confidence =  0.1 # 置信度

# 通过当前轨迹的cam和最后一帧zone_num，找到其应当存在哪个cam的哪个区域的待匹配队列中
next_cams_zone = {'c001':{1:[['c002',1],['c003',1],['c004',1],['c005',1]], 
                          2:[['c005',2]], 
                          3:[], 
                          4:[['c002',3]]},
                  'c002':{1:[['c001',1],['c003',1],['c004',1],['c005',1]], 
                          2:[['c006',3]],
                          3:[['c001',4],['c007',3]], 
                          4:[['c003',3]]},
                  'c003':{1:[['c001',1],['c002',1],['c004',1],['c005',1]], 
                          2:[['c006',1]], 
                          3:[['c002',4]], 
                          4:[['c004',3]]},
                  'c004':{1:[['c001',1],['c002',1],['c003',1],['c005',1]], 
                          2:[['c006',4]], 
                          3:[['c003',4]], 
                          4:[['c005',3]]},
                  'c005':{1:[['c001',1],['c002',1],['c003',1],['c004',1]], 
                          2:[['c001',2]], 
                          3:[['c007',4],['c004',4]], 
                          4:[]},
                  'c006':{1:[['c003',2]], 
                          2:[['c007',1]], 
                          3:[['c002',2]], 
                          4:[['c004',2]]},
                  'c007':{1:[['c006',2]], 
                          2:[], 
                          3:[['c001',4]], 
                          4:[['c004',4]]}}
# key1摄像头, key2为start_zone, 即该轨迹只能从这个zone邻接的摄像头中寻找匹配轨迹
# 1 白色 2 红色 3 绿色 4 蓝色

# 通过当前轨迹的起始zone, 定位到需要进行跨视频匹配的轨迹列表
# 再基于out_time二分查找，快速定位到最可能匹配的位置，向两侧匹配，直至超过规定值
Track_to_be_matched = {'c001':{1:[], 2:[], 3:[], 4:[]},
                       'c002':{1:[], 2:[], 3:[], 4:[]},
                       'c003':{1:[], 2:[], 3:[], 4:[]},
                       'c004':{1:[], 2:[], 3:[], 4:[]},
                       'c005':{1:[], 2:[], 3:[], 4:[]},
                       'c006':{1:[], 2:[], 3:[], 4:[]},
                       'c007':{1:[], 2:[], 3:[], 4:[]}}
# c001 4表示 区域4邻接的摄像头到c001的时间差，即c002的区域3的out_time - c001的区域4的in_time
avg_times = {'c001':{1:60.0, 2:50.0, 4:22.8},
             'c002':{1:60.0, 2:15.0, 3:15.0, 4:49.3},
             'c003':{1:60.0, 2:1.0, 3:52.6, 4:27.8},
             'c004':{1:60.0, 2:15.0, 3:50.3, 4:2.8},
             'c005':{1:60.0, 2:50.0, 3:33.4},
             'c006':{1:1.0, 2:1.0, 3:20.0, 4:20.0},
             'c007':{1:1.0, 2:0.0, 3:15.0, 4:15.0}}



def get_sim_matrix(_cfg,cid_tid_dict,cid_tids):
    count = len(cid_tids)
    print('count: ', count)

    q_arr = np.array([cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    g_arr = np.array([cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    q_arr = normalize(q_arr, axis=1)
    g_arr = normalize(g_arr, axis=1)
    # sim_matrix = np.matmul(q_arr, g_arr.T)

    # st mask
    st_mask = np.ones((count, count), dtype=np.float32)
    st_mask = intracam_ignore(st_mask, cid_tids)
    st_mask = st_filter(st_mask, cid_tids, cid_tid_dict)

    # visual rerank
    visual_sim_matrix = visual_rerank(q_arr, g_arr, cid_tids, _cfg)
    visual_sim_matrix = visual_sim_matrix.astype('float32')
    print(visual_sim_matrix)
    # merge result
    np.set_printoptions(precision=3)
    sim_matrix = visual_sim_matrix * st_mask

    # sim_matrix[sim_matrix < 0] = 0
    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix

def normalize(nparray, axis=0):
    nparray = preprocessing.normalize(nparray, norm='l2', axis=axis)
    return nparray

def get_match(cluster_labels):
    cluster_dict = dict()
    cluster = list()
    for i, l in enumerate(cluster_labels):
        if l in list(cluster_dict.keys()):
            cluster_dict[l].append(i)
        else:
            cluster_dict[l] = [i]
    for idx in cluster_dict:
        cluster.append(cluster_dict[idx])
    return cluster

def get_cid_tid(cluster_labels,cid_tids):
    cluster = list()
    for labels in cluster_labels:
        cid_tid_list = list()
        for label in labels:
            cid_tid_list.append(cid_tids[label])
        cluster.append(cid_tid_list)
    return cluster

def combin_cluster(sub_labels,cid_tids):
    cluster = list()
    for sub_c_to_c in sub_labels:
        if len(cluster)<1:
            cluster = sub_labels[sub_c_to_c]
            continue

        for c_ts in sub_labels[sub_c_to_c]:
            is_add = False
            for i_c, c_set in enumerate(cluster):
                if len(set(c_ts) & set(c_set))>0:
                    new_list = list(set(c_ts) | set(c_set))
                    cluster[i_c] = new_list
                    is_add = True
                    break
            if not is_add:
                cluster.append(c_ts)
    labels = list()
    num_tr = 0
    for c_ts in cluster:
        label_list = list()
        for c_t in c_ts:
            label_list.append(cid_tids.index(c_t))
            num_tr+=1
        label_list.sort()
        labels.append(label_list)
    print("new tricklets:{}".format(num_tr))
    return labels,cluster

def combin_feature(cid_tid_dict,sub_cluster):
    for sub_ct in sub_cluster:
        if len(sub_ct)<2: continue
        mean_feat = np.array([cid_tid_dict[i]['mean_feat'] for i in sub_ct])
        for i in sub_ct:
            cid_tid_dict[i]['mean_feat'] = mean_feat.mean(axis=0)
    return cid_tid_dict

def get_labels(_cfg, cid_tid_dict, cid_tids, score_thr):
    # 1st cluster
    sub_cid_tids = subcam_list(cid_tid_dict,cid_tids)
    sub_labels = dict()
    dis_thrs = [0.7,0.5,0.5,0.5,0.5,
                0.7,0.5,0.5,0.5,0.5]
    for i,sub_c_to_c in enumerate(sub_cid_tids):
        sim_matrix = get_sim_matrix(_cfg,cid_tid_dict,sub_cid_tids[sub_c_to_c])
        cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1-dis_thrs[i], affinity='precomputed',
                                linkage='complete').fit_predict(1 - sim_matrix)
        labels = get_match(cluster_labels)
        cluster_cid_tids = get_cid_tid(labels,sub_cid_tids[sub_c_to_c])
        sub_labels[sub_c_to_c] = cluster_cid_tids
    print("old tricklets:{}".format(len(cid_tids)))
    labels,sub_cluster = combin_cluster(sub_labels,cid_tids)

    # 2ed cluster
    cid_tid_dict_new = combin_feature(cid_tid_dict, sub_cluster)
    sub_cid_tids = subcam_list2(cid_tid_dict_new,cid_tids)
    sub_labels = dict()
    for i,sub_c_to_c in enumerate(sub_cid_tids):
        sim_matrix = get_sim_matrix(_cfg,cid_tid_dict_new,sub_cid_tids[sub_c_to_c])
        cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1-0.1, affinity='precomputed',
                                linkage='complete').fit_predict(1 - sim_matrix)
        labels = get_match(cluster_labels)
        cluster_cid_tids = get_cid_tid(labels,sub_cid_tids[sub_c_to_c])
        sub_labels[sub_c_to_c] = cluster_cid_tids
    print("old tricklets:{}".format(len(cid_tids)))
    labels,sub_cluster = combin_cluster(sub_labels,cid_tids)

    # 3rd cluster
    # cid_tid_dict_new = combin_feature(cid_tid_dict,sub_cluster)
    # sim_matrix = get_sim_matrix(_cfg,cid_tid_dict_new, cid_tids)
    # cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - 0.2, affinity='precomputed',
    #                                          linkage='complete').fit_predict(1 - sim_matrix)
    # labels = get_match(cluster_labels)
    return labels

def cfg_extract():
    cfg = CfgNode()
    cfg.REID_MODEL= 'detector/yolov5/reid/reid_model/resnet101_ibn_a_2.pth'
    cfg.REID_BACKBONE= 'resnet101_ibn_a'
    cfg.REID_SIZE_TEST= [384, 384]
    cfg.freeze()
    return cfg

extract_cfg = cfg_extract()

def read_data_from_txt(file_path, cams_ratio): 
    # 抽帧，有选择的加载gt文件中的行
    data_dict = {} 
    with open(file_path, 'r') as file:
        for line in file:
            # 从每一行提取tid、fid和bbox
            fid,tid,x1,y1,w,h,_,_,_,_ = map(int, line.strip().split(','))
            data = [tid, x1, y1, w, h]

            if fid % cams_ratio != 0:
                continue
            if fid not in data_dict:
                data_dict[fid] = {'gt':[],'detections':[]} 
            data_dict[fid]['gt'].append(data)

    return data_dict

def calculate_iou(bbox1, bbox2):
    # 计算两个矩形框的交并比
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = bbox1[2] * bbox1[3]
    bbox2_area = bbox2[2] * bbox2[3]
    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def find_overlap(interval1, interval2):
    start_point = max(interval1[0], interval2[0])
    end_point = min(interval1[1], interval2[1])

    if start_point <= end_point:
        return start_point, end_point
    else:
        return None  # 无重叠部分

def filter_res(data):
    # 过滤可能的重复项
    max_iou_dict = {}
    for item in data:
        ts_tid, gt_tid, start_point, end_point, frame_cnt, avg_iou = item
        avg_iou = float(avg_iou)  # 将avg_iou转换为浮点数
        if gt_tid not in max_iou_dict:
            max_iou_dict[gt_tid] = {'data': item, 'max_iou': avg_iou}
        else:
            if avg_iou > max_iou_dict[gt_tid]['max_iou']:
                max_iou_dict[gt_tid] = {'data': item, 'max_iou': avg_iou}
    filtered_data = [max_iou_dict[key]['data'] for key in max_iou_dict]
    return filtered_data

def mot_metrics(dict1, dict2, iou_threshold=0.5): # 输入是检测结果和gt
    match_tids = [['ts_tid','gt_tid', 'start_point', 'end_point', 'frame_cnt' 'avg_iou']]
    for tid1 in dict1.keys():
        avg_iou = iou_threshold
        match_tuple = []
        for tid2 in dict2.keys():
            f_list1 = list(dict1[tid1].keys())
            f_list2 = list(dict2[tid2].keys())
            overlap = find_overlap([f_list1[0],f_list1[-1]], [f_list2[0],f_list2[-1]])
            frame_cnt = 0
            if overlap:
                total_iou = 0
                start_point, end_point = overlap
                for i in range(start_point, end_point + 1):
                    if (i in f_list1) and (i in f_list2):
                        total_iou += calculate_iou(dict1[tid1][i], dict2[tid2][i])
                        frame_cnt += 1
                if (total_iou / (end_point - start_point + 1)) > avg_iou:
                    avg_iou = total_iou / frame_cnt
                    match_tuple = [tid1, tid2, start_point, end_point, frame_cnt, round(avg_iou,5)]
        # 找到平均iou最大的
        if len(match_tuple) > 0:
            if (match_tuple[1] == match_tids[-1][1]):
                if (match_tuple[-1] > match_tids[-1][-1]):
                    match_tids.pop()
                    match_tids.append(match_tuple)
            else:
                match_tids.append(match_tuple)
    match_res = filter_res(match_tids[1:])
    # 统计评估指标
    true_positive = 0
    d1_len = 0
    d2_len = 0
    for x in match_res:
        true_positive += int(x[4])
    for _,v in dict1.items():
        d1_len += len(v)
    for _,v in dict2.items():
        d2_len += len(v)
    false_positive = d1_len - true_positive
    false_negative = d2_len - true_positive
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (false_negative + true_positive)
    f1_score = 200 * recall * precision / (precision + recall)
    return precision, recall, f1_score

def add_zone_num(lines,zones):
    mot_list = dict()
    for line in lines:
        fid = int(lines[line]['frame'][3:]) # 
        tid = lines[line]['id']
        bbox = list(map(lambda x:int(float(x)), lines[line]['bbox']))
        if tid not in mot_list:
            mot_list[tid] = dict()
        out_dict = lines[line]
        out_dict['zone'] = zones.get_zone(bbox) # 给bbox分配了zone_num
        mot_list[tid][fid] = out_dict # 字典
    return mot_list # 字典

def gather_sequence_info(det_feat_dic): # detection_file存的是det_feat.pkl的路径，改成存有feat的字典feat_dict
    min_frame_idx = 0
    max_frame_idx = 2000    
    update_ms = None
    feature_dim = 2048 # default
    bbox_dic = {}
    feat_dic = {}
    for image_name in det_feat_dic:
        # 获取帧idx
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
        # "groundtruth": groundtruth,
        # "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms,
        "frame_rate": 10 # 帧率
         # "frame_rate": int(info_dict["frameRate"])
    }
    return seq_info

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

# 更换视频源时，修改视频路径，配置zone分区图和roi图即可
# weights = 'yolov5s.pt' # 模型大小
# imgsz = 1280 # default是640，传入1280 [320, 640, 1280]
# cams_ratio的长度应与摄像头个数相同, 抽帧间隔，默认连续取1，可设置为1, 2, 3
def run_mtmc(target_cam_index = 0,
             cams_dir = 'datasets/AIC22_Track1_MTMC_Tracking/train/S10',
             weights = ['yolov5s.pt'], 
             imgsz = [1280], 
             cams_ratio = [1]):
    g_tid = 0
    # 加载模型
    device = select_device(str(GPU_ID))
    # 传入的weights是权重名的list，则返回的det_model为一一对应的权重模型的list
    det_model = attempt_load(weights, map_location=device)
    det_model.half() # to FP16
    # 重识别模型
    ext_model = ReidFeature(GPU_ID, extract_cfg)

    # 加载数据
    cams = os.listdir(cams_dir)
    cams.sort()

    stride = int(det_model.stride.max()) # stride = 32

    trackers_avg_feat = dict() # key是tid，value是该轨迹的相关信息
    datasets = dict()           # 帧迭代器
    trackers = dict()            # 追踪器
    results = dict()             # key是cam，value是list，用于保存追踪器结果
    gt_detect = dict()           # 用于保存检测输出的gt
    frame_nums = dict()
    mm_data = dict()

    for cam_idx,cam in enumerate(cams):
        if cam_idx != target_cam_index:
                continue
        video_dir = os.path.join(cams_dir, cam) + '/vdo.mp4'
        gt_dir = os.path.join(cams_dir, cam) + '/gt/gt.txt'
        gt_detect[cam] = {}
        mm_data[cam] = read_data_from_txt(gt_dir, cams_ratio[0])
        frame_nums[cam] = []
        datasets[cam] = LoadImages(video_dir, img_size=imgsz[0], stride=stride)
        results[cam] = {}
        trackers[cam] = JDETracker(min_confidence, vdo_frame_ratio / cams_ratio[0])
        # 创建模型的实例,初始化模型的权重
        det_model(torch.zeros(1, 3, imgsz[0], imgsz[0]).to(device).type_as(next(det_model.parameters())))

    # names存的是目标检测结果的种类，检测[2,5,7](car,bus,truck)
    names = det_model.module.names if hasattr(det_model, 'module') else det_model.names
    # 控制遍历帧数
    frame_cnt = 0
    # 时间偏置, 读取自timestamp
    time_bias = 0.0
    # scene_name为场景名，对应zone目录下文件夹
    zones = zone(scene_name=cams_dir[-3:])
    # 统计时间
    total_detect_time = 0
    total_extract_time = 0
    total_sct_time = 0
    total_pp_time = 0
    
    while True:
        # 处理目标摄像头的每一帧
        for cam_idx,cam in enumerate(cams):
            if cam_idx != target_cam_index:
                continue
            current_dict = dict()
            # 保存crop后的图像
            current_image_dict = dict()
            for path, img, im0s, vid_cap in datasets[cam]:
                if (getattr(datasets[cam], 'frame', 0) % cams_ratio[0] != 0):
                    break
                # 格式化img
                img = torch.from_numpy(img).to(device)
                img = img.half()
                img /= 255.0
                # 确保维度正确
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # 开始推理
                t1 = time_synchronized()
                # pred 是一个list
                pred = det_model(img, augment=False)[0]
                # 去除检测结果中冗余的边界框
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes=[2, 5, 7], agnostic=True)
                # 处理推理结果
                for i, det in enumerate(pred):
                    # 传入的是图片，dataset没有frame属性，frame=0
                    # 传入的是视频，则frame是当前帧数
                    p, s, im0, frame_idx = path, '', im0s, getattr(datasets[cam], 'frame', 0)
                    p = Path(p)
                    s += '%gx%g ' % img.shape[2:]
                    # 创建一个包含图像尺寸信息的 PyTorch 张量 
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    # print("det 的长度为 %d" % len(det))
                    # print("det 的类型为 %s" % type(det)) <class 'torch.Tensor'>
                    if len(det):
                        img_det = np.copy(im0)
                        # print("rescale前的det{}".format(det))
                        # Rescale boxes from img_size to im0 size (缩放边框)
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        det_num = 0 #  局部id
                        for *xyxy, conf, cls in reversed(det):
                            x1,y1,x2,y2 = tuple(torch.tensor(xyxy).view(4).tolist())
                            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                            # shape返回 [高度, 宽度, 通道数]
                            if x1 < 0 or y1 < 0 or x2 > im0.shape[1]-1  or y2 > im0.shape[0]-1:
                                # print('clip bbox')
                                continue
                            #if (y2-y1) * (x2-x1) < 1000:
                            if (y2-y1) < 32 or (x2-x1) < 32:
                                # print('det too small')
                                continue
                            if True:
                                det_name = "{}_{}_{:0>3d}".format(cam, frame_idx, det_num)
                                det_class = int(cls.tolist())
                                det_conf = conf.tolist()
                                current_image_dict[det_name] = img_det[y1:y2,x1:x2]
                                current_dict[det_name] = {
                                    'bbox': (x1,y1,x2,y2),
                                    'frame': frame_idx,
                                    'id': det_num,
                                    'imgname': det_name+".png",
                                    'class': det_class,
                                    'conf': det_conf
                                }
                            det_num += 1
                t2 = time_synchronized()
                total_detect_time += (t2 - t1)
                # 记录每帧的目标数
                # frame_nums[cam].append([frame_idx, det_num])
                break
            # 完成某个单视频的第n帧检测
            # current_image_dict中存有当前帧的车辆图片信息
            # current_dict中存有当前帧的bbox等信息
            if len(current_dict) == 0: # 未检测到车辆，跳过后续步骤
                continue

            # 单视频特征提取
            t3 = time_synchronized()
            reid_feat_numpy = ext_model.extract(current_image_dict)
            # 用于保存提取出来的特征
            current_feat_dict = dict()
            for index, ext_img in enumerate(current_image_dict.keys()):
                current_feat_dict[ext_img] = reid_feat_numpy[index]
            cur_det_feat_dict = current_dict.copy()
            for det_name, _ in current_dict.items():
                cur_det_feat_dict[det_name]['feat'] = current_feat_dict[det_name]
            t4 = time_synchronized()
            total_extract_time += (t4 - t3)

            # 单视频追踪
            t5 = time_synchronized()
            seq_info = gather_sequence_info(cur_det_feat_dict)          
            [bbox_dic, feat_dic] = seq_info['detections']
            if frame_idx not in bbox_dic:
                print(f'empty for {cam} {frame_idx}')
            detections = bbox_dic[frame_idx]
            feats = feat_dic[frame_idx]
            # Run non-maxima suppression.
            boxes = np.array([d[:4] for d in detections], dtype=float)
            scores = np.array([d[4] for d in detections], dtype=float)
            nms_keep = nms(torch.from_numpy(boxes),
                                    torch.from_numpy(scores),
                                    iou_threshold=0.99).numpy()
            detections = np.array([detections[i] for i in nms_keep], dtype=float)
            feats = np.array([feats[i] for i in nms_keep], dtype=float)

            # 更新对应的tracker (JDETracker目标追踪器)
            online_targets = trackers[cam].update(detections, feats, frame_idx)
            # online_targets = trackers[cam].update_without_embedding(detections, feats, frame_idx)
            # 更新对应的result
            for t in online_targets:
                tlwh = t.det_tlwh
                tid = t.track_id
                score = t.score
                feature = t.features[-1]
                feature = t.smooth_feat
                image_name = f'{cam}_{tid}_{frame_idx}'
                if tlwh[2] * tlwh[3] > 750:
                    if tid not in results[cam]:
                        results[cam][tid] = [[frame_idx, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3], feature]]
                    else:
                        results[cam][tid].append([frame_idx, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3], feature])
            t6 = time_synchronized()
            total_sct_time += (t6 - t5)

            # 将results变为np.array, 并append(feat), 维度调整为2054(6+ 2048)
            # 准备后处理数据
            # 仅对20~200区间的轨迹进行后处理
            pp_result = []
            for _, res_list in results[cam].items():
                # NOTE 10 与抽帧频率有关
                if frame_idx - res_list[-1][0] > 100 or frame_idx - res_list[-1][0] < 10:
                    continue
                for row in res_list:
                    [fid, tid, x, y, w, h] = row[:6]
                    feat = row[-1]
                    dummpy_input = np.array([fid, tid, x, y, w, h])
                    dummpy_input = np.concatenate((dummpy_input, feat))
                    pp_result.append(dummpy_input)
                    
            t7 = time_synchronized()
            pp_result = np.array(pp_result)
            if len(pp_result) == 0:
                t8 = time_synchronized()
                total_pp_time += (t8 - t7)
                continue
             # 执行后处理, 6表示feat的起始列
            pp_result, matches = associate(pp_result, 0.1, 6)
            # 使用更早的tid，将关联的轨迹对整合起来，修改results
            for new_tid, old_tid in matches.items():
                old_tra = results[cam][old_tid]
                new_tra = results[cam][new_tid]
                for row in new_tra:
                    row[1] = old_tid
                results[cam][old_tid] = old_tra + new_tra
                del results[cam][new_tid]
            # pp_result = track_nms(pp_result, 0.65)

            t8 = time_synchronized()
            total_pp_time += (t8 - t7)
                
            cid = int(cam[-3:])
            zones.set_cam(cid)
            mot_feat_dic = {}
            # TODO 修改zone部分数据结构
            for row in pp_result:
                [fid, tid, x, y, w, h] = row[:6]
                fid = int(fid)
                tid = int(tid)
                feat = np.array(row[-2048:])
                image_name = f'{cam}_{tid}_{fid}.png'
                bbox = (x, y, x+w, y+h)
                frame = f'img{int(fid):06d}'
                mot_feat_dic[image_name] = {'bbox': bbox, 'frame': frame, 'id': tid,
                                            'imgname': image_name, 'feat': feat}

            # 为bbox分配zone_num, 1 白色 2 红色 3 绿色 4 蓝色
            mot_list = add_zone_num(mot_feat_dic, zones)
            # 基于时间间隔和区域切分tracklet，切分间隔过久的和出现反向移动的轨迹 print(new old)
            mot_list = zones.break_mot(mot_list, cid)
            # 基于区域过滤tracklet
            # mot_list = zones.filter_mot(mot_list, cid)
            # 基于bbox过滤tracklet
            mot_list = zones.filter_bbox(mot_list, cid)

            for tid in mot_list:
                tracklet = mot_list[tid]
                if (len(tracklet)) <= 1: continue
                frame_list = list(tracklet.keys())
                frame_list.sort()
                # 遍历mot_list，若某tracklet的fid满足frame_idx - tracklet_fid > 20，延迟2s,则认为该轨迹已离开摄像区域
                if ((frame_idx - frame_list[-1] > 20) and (tid not in trackers_avg_feat)):
                    zone_list = [tracklet[f]['zone'] for f in frame_list]
                    feature_list = [tracklet[f]['feat'] for f in frame_list if (tracklet[f]['bbox'][3]-tracklet[f]['bbox'][1])*(tracklet[f]['bbox'][2]-tracklet[f]['bbox'][0])>2000]
                    if len(feature_list)<2:
                        feature_list = [tracklet[f]['feat'] for f in frame_list]
                    # 计算进出时间
                    io_time = [time_bias + frame_list[0] / 10., time_bias + frame_list[-1] / 10.]
                    # 计算轨迹的平均特征向量
                    all_feat = np.array([feat for feat in feature_list])
                    mean_feat = np.mean(all_feat, axis=0)
                    g_tid += 1
                    gt_detect[cam][g_tid] = dict()
                    for i in frame_list: 
                        x, y, w, h = map(int, [mot_list[tid][i]['bbox'][0],mot_list[tid][i]['bbox'][1],
                                               mot_list[tid][i]['bbox'][2] - mot_list[tid][i]['bbox'][0],
                                               mot_list[tid][i]['bbox'][3] - mot_list[tid][i]['bbox'][1]])
                        mm_data[cam][i]['detections'].append([g_tid,x,y,w,h])   
                        gt_detect[cam][g_tid][i] = [x,y,w,h]
                    
                    trackers_avg_feat[tid] = {
                        'g_tid' : g_tid,
                        'io_time': io_time,
                        'zone_list': zone_list,
                        'frame_list': frame_list,
                        'mean_feat': mean_feat,
                        'cam':cam,
                        'tid':tid,
                        'tracklet': tracklet
                    }
            t8 = time_synchronized()
            total_pp_time += (t8 - t7)

        # 跳出while循环, 设置为2000时，可以跑完整个视频
        if frame_cnt == 10010:
            break
        frame_cnt += 1
        # end for cam in cams 
    # end while
    print('done')

    for cam_idx,cam in enumerate(cams):
        if cam_idx != target_cam_index:
            continue
        if not os.path.exists(cams_dir[-3:]):
            os.makedirs(cams_dir[-3:])
        gt_write = os.path.join(cams_dir[-3:], cam) + "_gt_test.txt"
        detnum_write = os.path.join(cams_dir[-3:], cam) + "_detnum.txt"
        with open(gt_write, "w") as gt_file:
            for gid, v in gt_detect[cam].items():
                for fid, bbox in v.items():
                    gt_file.write(f'{fid},{gid},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},1,3,-1,-1\n')

        with open(detnum_write, "w") as detnum_file:
            # 写每帧检测目标的数量
            for row in frame_nums[cam]:
                detnum_file.write(f'{row[0]},{row[1]}\n')
    
    # 返回检测结果及各阶段耗时            
    return mm_data, total_detect_time, total_extract_time, total_sct_time, total_pp_time


def profile(target_cam_, cams_dir_, weights_, imgsz_, cams_ratio_, gpu_id):
    print("传过来的参数:", target_cam_, cams_dir_, weights_, imgsz_, cams_ratio_, gpu_id)
    global GPU_ID
    GPU_ID = gpu_id
    mm_data, total_detect_time, total_extract_time, total_sct_time, total_pp_time=run_mtmc(target_cam_index=target_cam_, cams_dir=cams_dir_, weights=weights_, imgsz=imgsz_, cams_ratio=cams_ratio_)
    processing_time = total_detect_time+total_extract_time+total_sct_time+total_pp_time
    cams = os.listdir(cams_dir_)
    cams.sort()
    
    for i, cam in enumerate(cams):
        if i != target_cam_:
            continue
        acc = mm.MOTAccumulator(auto_id=True)
        for frame, frame_data in mm_data[cam].items():
            gt = frame_data['gt']
            detections = frame_data['detections']
            # 提取gt轨迹id和bbox框
            gt_ids = [item[0] for item in gt]
            gt_bboxes = [item[1:] for item in gt]
            # 提取检测结果id和bbox框
            detection_ids = [item[0] for item in detections]
            detection_bboxes = [item[1:] for item in detections]
            dists = mm.distances.iou_matrix(gt_bboxes, detection_bboxes, max_iou=0.5)
            acc.update(gt_ids, detection_ids, dists)
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'precision', 'recall'], name='acc')
        precision, recall = summary['precision'][0], summary['recall'][0]
        f1 = 2 * recall * precision / (precision + recall)   
    print("准确率跟处理时间:", f1, processing_time)
    result_json = dict()
    result_json['precision'] = precision
    result_json['recall'] = recall
    result_json['total_detect_time'] = total_detect_time
    result_json['total_extract_time'] = total_extract_time
    result_json['total_sct_time'] = total_sct_time
    result_json['total_pp_time'] = total_pp_time
    return f1, processing_time, result_json

# if __name__ == '__main__':
#     with torch.no_grad():
#         run_mtmc()
#         # pass