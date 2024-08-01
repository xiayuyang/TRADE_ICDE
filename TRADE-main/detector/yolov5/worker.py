import os
import sys
import queue
import logging
import traceback
import multiprocessing as mp
from copy import deepcopy
from multiprocessing.managers import SyncManager
import time
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from numpy import random

from torchvision.ops import nms

from reid.matching.tools.utils.filter import *
from reid.matching.tools.utils.zone_intra import zone

from MOTBaseline.src.fm_tracker.multitracker import JDETracker
from MOTBaseline.src.post_processing.post_association import associate
from MOTBaseline.src.post_processing.track_nms import track_nms

from models.experimental import attempt_load
from multi_utils import (
    add_zone_num,
    gather_sequence_info,
    scale_coords,
    ReidFeature,
    cfg_extract,
)
from utils.datasets import LoadStreams, LoadImages
from utils.general import non_max_suppression

from utils.plots import plot_one_box
from utils.torch_utils import select_device

CAM_NUMBER = 4
FQ_SIZE = 1
TQ_SIZE = 10
TOTAL_FRAME = 10000 # 需要运行多少帧
CONF_THRES = 0.25
IOU_THRES = 0.45
MIN_CONFIDENCE = 0.1
FRAME_RATE = 20
PP_THRES = 20

# 绿3 蓝4
next_cams_zone = {'c001':{3:[['c004',4]], 
                          4:[['c003',3]]},
                  'c002':{3:[], 
                          4:[['c004',3]]},
                  'c003':{3:[['c001',4]], 
                          4:[]},
                  'c004':{3:[['c002',4]], 
                          4:[['c001',3]]}}
                  
Track_to_be_matched = {'c001':{3:[], 4:[]},
                       'c002':{3:[], 4:[]},
                       'c003':{3:[], 4:[]},
                       'c004':{3:[], 4:[]}}

avg_times = {'c001':{3:30.6, 4:30.2},
             'c002':{3:0.0, 4:32.2},
             'c003':{3:30.2, 4:0.0},
             'c004':{3:32.2, 4:30.6}}

frame_nums = dict()
trackers = dict()
results = dict()
trackers_avg_feat = dict()

def cal_similarity(vector1,vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

def find_closest_element(data, target_time):
    low, high = 0, len(data) - 1
    
    while low <= high:
        mid = (low + high) // 2
        mid_time = data[mid][0]

        if mid_time < target_time:
            low = mid + 1
        elif mid_time > target_time:
            high = mid - 1
        else:
            return mid

    if low > 0 and (high == len(data) - 1 or abs(data[low][0] - target_time) < abs(data[high][0] - target_time)):
        return low
    else:
        return high

def cross_cam_match(cam, start_zone, io_time, new_feat, tid, max_similarity = 0.7):
    global g_tid
    pre_similarity = 0.5
    match_flag = False
    match_list = Track_to_be_matched[cam][start_zone]
    match_list_size = len(match_list)
    match_idx = -1
    if match_list_size == 0:
        g_tid += 1
        return g_tid
    out_time = io_time[0]-avg_times[cam][start_zone]
    closest_idx = find_closest_element(match_list, out_time)
    closest_idx = max(0, min(closest_idx, match_list_size - 1))
    max_range = max(closest_idx + 1, match_list_size - closest_idx)
    for search_idx in range(2 * max_range):
        cur_idx = (closest_idx + search_idx // 2) if search_idx % 2 == 0 else (closest_idx - 1 - search_idx // 2 )
        if (cur_idx >= 0 and cur_idx <= (match_list_size - 1)):
            cur_similarity = cal_similarity(new_feat, match_list[cur_idx][2])
            # 可能匹配轨迹格式[out_time, g_tid, mean_feat, is_matched, similarity, tid]
            if cur_similarity > match_list[cur_idx][4] and cur_similarity > pre_similarity:
                match_flag  = True
                pre_similarity = cur_similarity
                match_idx = cur_idx
                if cur_similarity > max_similarity:
                    break
    if match_flag:       
        if match_list[match_idx][3]:
            # 处理冲突
            conflict_tid = match_list[match_idx][-1]
            Track_to_be_matched[cam][start_zone][match_idx][5] = tid
            Track_to_be_matched[cam][start_zone][match_idx][3] = True
            Track_to_be_matched[cam][start_zone][match_idx][4] = pre_similarity
            conflict_iotime = trackers_avg_feat[conflict_tid]['io_time']
            conflict_feat = trackers_avg_feat[conflict_tid]['mean_feat']
            conflitct_gtid = cross_cam_match(cam, start_zone, conflict_iotime, conflict_feat, conflict_tid)
            trackers_avg_feat[cam][conflict_tid]['g_tid'] = conflitct_gtid
        return match_list[match_idx][1]
    else:
        g_tid += 1
        return g_tid           

def worker_cam(cam:str, frame_q:queue.Queue, traj_q:queue.Queue, 
               dataset, device, det_model, ext_model, zones):
    try:
        while(True):
            current_dict = dict() # 用于保存当前帧检测结果
            current_image_dict = dict() # 用于保存当前帧检测图像
            img, im0s = frame_q.get(block=True, timeout=None) # 获取视频帧
            img = torch.from_numpy(img).to(device)  # 格式化img
            img = img.half()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # 目标检测
            pred = det_model(img, augment=False)[0]
            pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=[2, 5, 7], agnostic=True)
            for i, det in enumerate(pred):
                frame_idx = getattr(dataset, 'frame', 0)
                if len(det):
                    img_det = np.copy(im0s)
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                    det_num = 0 # 局部id
                    for *xyxy, conf, cls in reversed(det):
                        x1,y1,x2,y2 = tuple(torch.tensor(xyxy).view(4).tolist())
                        if x1 < 0 or y1 < 0 or x2 > im0s.shape[1]-1  or y2 > im0s.shape[0]-1:
                            continue
                        if (y2-y1) < 32 or (x2-x1) < 32: # 过滤过小的检测框
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
            frame_nums[cam].append([frame_idx, det_num]) # 记录每帧的目标数
            if len(current_dict) == 0: # 未检测到车辆，跳过后续步骤
                continue
            
            # 特征提取
            reid_feat_numpy = ext_model.extract(current_image_dict)
            current_feat_dict = {}
            for index, ext_img in enumerate(current_image_dict.keys()):
                current_feat_dict[ext_img] = reid_feat_numpy[index]
            cur_det_feat_dict = current_dict.copy()
            for det_name, _ in current_dict.items():
                cur_det_feat_dict[det_name]['feat'] = current_feat_dict[det_name]

            # 单视频追踪
            seq_info = gather_sequence_info(cur_det_feat_dict)          
            [bbox_dic, feat_dic] = seq_info['detections']
            if frame_idx not in bbox_dic:
                print(f'empty for {cam} {frame_idx}')
            detections = bbox_dic[frame_idx]
            feats = feat_dic[frame_idx]
            boxes = np.array([d[:4] for d in detections], dtype=float)
            scores = np.array([d[4] for d in detections], dtype=float)
            nms_keep = nms(torch.from_numpy(boxes),
                                    torch.from_numpy(scores),
                                    iou_threshold=0.99).numpy()
            detections = np.array([detections[i] for i in nms_keep], dtype=float)
            feats = np.array([feats[i] for i in nms_keep], dtype=float)
            online_targets = trackers[cam].update(detections, feats, frame_idx)
            # online_targets = trackers[cam].update_without_embedding(detections, feats, frame_idx)
            for t in online_targets: # 更新result
                tlwh = t.det_tlwh
                tid = t.track_id
                feature = t.features[-1]
                feature = t.smooth_feat
                if tlwh[2] * tlwh[3] > 750:
                    if tid not in results[cam]:
                        results[cam][tid] = [[frame_idx, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3], feature]]
                    else:
                        results[cam][tid].append([frame_idx, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3], feature])
            
            # 后处理
            pp_result = []
            for _, res_list in results[cam].items():
                if frame_idx - res_list[-1][0] > 100 or frame_idx - res_list[-1][0] < PP_THRES:
                    continue
                for row in res_list:
                    [fid, tid, x, y, w, h] = row[:6]
                    feat = row[-1]
                    dummpy_input = np.array([fid, tid, x, y, w, h])
                    dummpy_input = np.concatenate((dummpy_input, feat))
                    pp_result.append(dummpy_input)
            pp_result = np.array(pp_result)
            if len(pp_result) == 0:
                continue

            pp_result, matches = associate(pp_result, 0.1, 6)
            # 使用更早的tid，将关联的轨迹对整合起来，修改results
            for new_tid, old_tid in matches.items():
                old_tra = results[cam][old_tid]
                new_tra = results[cam][new_tid]
                for row in new_tra:
                    row[1] = old_tid
                results[cam][old_tid] = old_tra + new_tra
                del results[cam][new_tid]
            pp_result = track_nms(pp_result, 0.65)
              
            cid = int(cam[-3:])
            zones.set_cam(cid)
            mot_list = {}
            for row in pp_result:
                [fid, tid, x, y, w, h] = row[:6]
                fid = int(fid)
                tid = int(tid)
                feat = np.array(row[-2048:])
                bbox = (x, y, x+w, y+h)
                zone_num = zones.get_zone(bbox)
                mot_list[tid][fid] =  {'bbox':bbox,
                                       'zone':zone_num,
                                       'feat':feat}

            res_dict = dict()
            for tid in mot_list:
                tracklet = mot_list[tid]
                if (len(tracklet)) <= 1: continue
                frame_list = list(tracklet.keys())
                frame_list.sort()
                # 遍历mot_list，若某tracklet的fid满足frame_idx - tracklet_fid > 20，延迟2s,则认为该轨迹已离开摄像区域
                if ((frame_idx - frame_list[-1] > PP_THRES) and (tid not in trackers_avg_feat)):
                    bbox_list = [tracklet[f]['bbox'] for f in frame_list]
                    zone_list = [tracklet[f]['zone'] for f in frame_list]
                    feature_list = [tracklet[f]['feat'] for f in frame_list]
                    
                    if len(feature_list)<2:
                        feature_list = [tracklet[f]['feat'] for f in frame_list]
                    # 计算进出时间
                    io_time = [frame_list[0] / 10., frame_list[-1] / 10.]
                    # 计算轨迹的平均特征向量
                    all_feat = np.array([feat for feat in feature_list])
                    mean_feat = np.mean(all_feat, axis=0)
                    res_dict[tid] = {
                        'io_time': io_time,
                        'bbox_list':bbox_list,
                        'zone_list': zone_list,
                        'frame_list': frame_list,
                        'mean_feat': mean_feat,
                        'tracklet': tracklet
                    }
            # 将单视频轨迹字典传出
            traj_q.put(item=res_dict, block=True, timeout=None)

            # 退出工作进程
            if frame_idx == TOTAL_FRAME:
                break
    except Exception as e:
        print(e.args)
        print(traceback.format_exc())
    finally:
        print(f"CAM WORKER {id} PROCESS EXIT!\n")

def main(cams_dir = 'datasets/AIC22_Track1_MTMC_Tracking/train/S12',
         weights = ['yolov5s.pt','yolov5s.pt','yolov5s.pt','yolov5s.pt'], 
         imgszs = [1280,1280,1280,1280], 
         cams_ratio = [1, 1, 1, 1],
         lazy_threshold = 0.7):
    
    proc_list = list()
    manager = SyncManager()
    manager.start()
    frame_qs = [manager.Queue(maxsize=FQ_SIZE) for _ in range(CAM_NUMBER)]
    traj_qs = [manager.Queue(maxsize=TQ_SIZE) for _ in range(CAM_NUMBER)]

    cams = os.listdir(cams_dir)
    cams.sort()
    datasets = dict()
    gt_detect = dict()
    zones = zone(scene_name=cams_dir[-3:])
    frame_cnt = 0
    # 共享变量g_tid和进程锁，保证各进程对g_tid的修改是原子的
    # g_tid = mp.Value('i', 0)
    # gtid_lock = mp.Lock()

    try:
        for i,cam in enumerate(cams):
            # 初始化存储中间数据的结构
            frame_nums[cam] = list()
            results[cam] = dict()
            # 加载模型
            device = select_device(str(i % CAM_NUMBER))
            det_model = attempt_load(weights[i], map_location=device)
            det_model.half() # to FP16
            # 创建模型实例，初始化模型的权重
            det_model(torch.zeros(1, 3, imgszs[i], imgszs[i]).to(device).type_as(next(det_model.parameters())))
            stride = int(det_model.stride.max())
            # 特征提取配置
            extract_cfg = cfg_extract()
            # 加载重识别模型
            ext_model = ReidFeature(i % CAM_NUMBER, extract_cfg)
            # 初始化追踪器
            trackers[cam] = JDETracker(MIN_CONFIDENCE, FRAME_RATE // cams_ratio[i])
            
            # 初始化视频迭代器
            video_dir = os.path.join(cams_dir, cams[i]) + '/vdo.mp4'
            datasets[cam] = LoadImages(video_dir, img_size=imgszs[i], stride=stride)
            gt_detect[cam] = dict()
            # 创建工作进程
            cam_proc = mp.Process(target=worker_cam, 
                                  args= (cam, frame_qs[i], traj_qs[i], datasets[cam], device, det_model, ext_model, zones))
            # 启动进程
            cam_proc.start()
            proc_list.append(cam_proc)
        
        while(True):
            for i,cam in enumerate(cams):
                if (getattr(datasets[cam], 'frame', 0) % cams_ratio[i] != 0):
                    continue
                _, img, im0s, _ = next(datasets[cam])
                frame_qs[i].put(item=[img,im0s], block=True, timeout=None)

            for i,cam in enumerate(cams):
                traj_dict = traj_qs[i].get(block=True, timeout=3)
                for tid, t_val in traj_dict.items():
                    io_time = t_val['io_time']
                    zone_list = t_val['zone_list']
                    frame_list = t_val['frame_list']
                    bbox_list = t_val['bbox_list']
                    mean_feat = t_val['mean_feat']
                    tracklet = t_val['tracklet']
                    start_zone = zone_list[0]
                    end_zone = zone_list[-1]
                    next_area = next_cams_zone[cam][end_zone]

                    if start_zone: # 起始区域为0，则必为新轨迹
                        matched_tid = cross_cam_match(cam, start_zone, io_time, mean_feat, tid)
                        if end_zone:
                            next_area = next_cams_zone[cam][end_zone]
                    else:
                        g_tid += 1
                        matched_tid = g_tid
                    for next_cam, next_zone  in next_area:
                        # 可能匹配轨迹格式[out_time, g_tid, mean_feat, is_matched, similarity]
                        Track_to_be_matched[next_cam][next_zone].append([io_time[1], matched_tid, mean_feat, False, 0.5, tid])

                    gt_detect[cam][matched_tid] = dict()
                    for i,frame_idx in enumerate(frame_list):                  
                        gt_detect[cam][matched_tid][frame_idx] = [int(bbox_list[i][0]),int(bbox_list[i][1]),
                                                int(bbox_list[i][2] - bbox_list[i][0]), 
                                                int(bbox_list[i][3] - bbox_list[i][1])]

                    trackers_avg_feat[tid] = {
                        'g_tid' : matched_tid,
                        'io_time': io_time,
                        'zone_list': zone_list,
                        'frame_list': frame_list,
                        'mean_feat': mean_feat,
                        'cam':cam,
                        'tid':tid,
                        'tracklet': tracklet
                    }

            # 实际运行的帧数
            if frame_cnt == TOTAL_FRAME:
                break
            frame_cnt += 1

    except Exception as e:
        print(e.args)
        print(traceback.format_exc())
    finally:
        # 结束工作进程
        [p.join() for p in proc_list]
        manager.shutdown()
        print("DONE, EXIT MAIN PROCESS!")

if __name__ == '__main__':
    try:
        mp.set_start_method(method='spawn',force=True)  # force all the multiprocessing to 'spawn' methods
        main()
    finally:
        pass