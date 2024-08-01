import multiprocessing
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

def detect_objects(cam_idx, device):
    cams_dir = 'datasets/AIC22_Track1_MTMC_Tracking/train/S13'
    cams = os.listdir(cams_dir)
    cams.sort()
    cam = cams[cam_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = 'yolov5l.pt'
    imgsz = 1280
    conf_thres = 0.25
    iou_thres = 0.45
    results = []
    det_model = attempt_load(weights, map_location=device)
    det_model.half()
    stride = int(det_model.stride.max())
    video_dir = os.path.join(cams_dir, cam) + '/vdo.mp4'
    dataset = LoadImages(video_dir, img_size=1280, stride=stride)
    det_model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(det_model.parameters())))
    t0 = time.time()
    det_cnt = 0
    for path, img, im0s, vid_cap in dataset:
        t1 = time.time()
        if t1 - t0 > 100:
            break
        # img_det = copy.deepcopy(img)
        # 格式化img
        img = torch.from_numpy(img).to(device)
        img = img.half()
        img /= 255.0
        # 确保维度正确
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = det_model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=[2, 5, 7], agnostic=True)
        for i, det in enumerate(pred):
            p, s, im0, frame_idx = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)
            s += '%gx%g ' % img.shape[2:]
            if len(det):

                # Rescale boxes from img_size to im0 size (缩放边框)
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                for *xyxy, conf, cls in reversed(det):
                    x1,y1,x2,y2 = tuple(torch.tensor(xyxy).view(4).tolist())
                    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                    # scale_ratio = int(1280/imgsz[cam_idx])
                    # x1,y1,x2,y2 = int(x1//scale_ratio),int(y1//scale_ratio),int(x2//scale_ratio),int(y2//scale_ratio)
                    # shape返回 [高度, 宽度, 通道数]
                    if x1 < 0 or y1 < 0 or x2 > im0.shape[1]-1  or y2 > im0.shape[0]-1:
                        # print('clip bbox')
                        continue
                    #if (y2-y1) * (x2-x1) < 1000:    # 过滤过小的检测框
                    if (y2-y1) <= 32 or (x2-x1) <= 32 : # 1280 /1280
                        continue

        det_cnt += 1

    print(f'{cam} has detected {det_cnt}')


def main():
    cams_dir = 'datasets/AIC22_Track1_MTMC_Tracking/train/S13'
    GPU_ID = 0
    cams = os.listdir(cams_dir)
    cams.sort()
    cams_num = len(cams)
    
    # 检查CUDA是否可用
    device = select_device(str(GPU_ID))
    print(f"Using device: {device}")
    
    #results_queue = multiprocessing.Queue()
    ctx = multiprocessing.get_context("spawn")

    processes = []
    num_processes = 4  # 设置使用的进程数量

    for i in range(num_processes):
        cam_idx = i % cams_num
        p = ctx.Process(target=detect_objects, args=(cam_idx, device))
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()

    # 收集所有进程的结果
    # all_results = []
    # while not results_queue.empty():
    #     all_results.append(results_queue.get())

    # # 处理结果（这里仅仅打印结果）
    # for i, result in enumerate(all_results):
    #     print(f"Result from process {i+1}: {result}")

if __name__ == "__main__":
    main()
