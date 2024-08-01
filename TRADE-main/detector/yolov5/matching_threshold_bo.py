'''
    多视频的tuning, 单视频的旋钮都不懂。主要是看一下matching_threshold对结果的影响
'''

import numpy as np
import os
from openbox import space as sp
import sys
import os
import random
import queue
import logging
import traceback
import multiprocessing as mp
import numpy as numpy
from copy import deepcopy
from multiprocessing.managers import SyncManager
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
from utils.torch_utils import time_synchronized
import datetime 
import json
from openbox import Optimizer
# NOTE
from S13_mcmt import profile
# TODO 使用的GPU是哪个
os.environ["CUDA_VISIBLE_DEVICES"]="3"
GPU_ID = 3
TEST = False
# NOTE
cams_dir = 'datasets/AIC22_Track1_MTMC_Tracking/train/S13'
cams = os.listdir(cams_dir)
camera_num=len(cams)



current_date = datetime.date.today()
# 将日期格式化为字符串，格式为 YYYY-MM-DD
formatted_date = current_date.strftime('%Y-%m-%d')




# 1 hours ~ 9 hours
min_processing_time = 1*30*30
max_processing_time = 9*30*30
matching_low_bound = 0.5
matching_upper_bound = 0.9

# 创建文件名,存每次的config与objective值
config_file_name = f"./tune_result/{formatted_date}.json"
data = dict()
data['min_processing_time'] = min_processing_time
data['max_processing_time'] = max_processing_time
data['matching_low_bound'] = matching_low_bound

data['matching_upper_bound'] = matching_upper_bound
data['log'] = 'config_updated_pp normalize_F1 matching_crossmatch, analysis matching threshold impact on result and its comparision number'
# 写入 JSON 数据到文件
with open(config_file_name, 'w') as file:
    json.dump(data, file, indent=4)

# 可以考虑把processing_time与accuracy归一化一下, 
# 并且把accuracy取反一下，因为openbox默认是最小化目标函数
def multi_camera_matching(config: sp.Configuration):
    # convert Configuration into Python dict.
    params = config.get_dictionary().copy()
    # FPS_CHOICE = ["1", "2", "4", "10"]
    # RESOLUTION_CHOICE = ["320", "640", "1280"]
    # SIZE_CHOICE = ["yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]
    # 固定这些单视频的旋钮
    FPS_CHOICE = ["4"]
    RESOLUTION_CHOICE = ["1280"]
    SIZE_CHOICE = ["yolov5m.pt"]
    weights = []
    imgsz = []
    cams_ratio = []
    global GPU_ID
    for i in range(camera_num):
        weight_name = 'model_size_camera' + str(i+1)
        fps_name = 'fps_camera'+str(i+1)
        resolution_name = 'resolution_camera' + str(i+1)
        weights.append(params[weight_name])
        imgsz.append(int(params[resolution_name]))
        cams_ratio.append(int(params[fps_name]))
    lazy_threshold = params['lazy_threshold']
    with open(config_file_name, 'a') as file:
        json.dump(params, file)
        file.write('\n') 
    if TEST:
        result, result_json = normalize_objective(test=True)
    else:
        accuracy, processing_time, section_json = profile(
            cams_dir_= cams_dir, weights_= weights, imgsz_=imgsz, cams_ratio_=cams_ratio, lazy_threshold_=lazy_threshold, gpu_id = GPU_ID)
        print("accuracy跟processingtime等于", accuracy, processing_time)
        result, result_json = normalize_objective(accuracy, processing_time, test=False)
        result_json['raw_accuracy'] = accuracy
        result_json['raw_processing_time'] = processing_time
    # print("config与result等于:", params, result)
    with open(config_file_name, 'a') as file:
        json.dump(result_json, file)
        file.write('\n')
        json.dump(section_json, file)
        file.write('\n')
    return result


def normalize_objective(accuracy=0, processing_time=0, test=False):
    result = dict()
    result_json = dict()
    if test:
        result['objectives'] = np.stack([random.random(), random.random()], axis=-1)
        result_json['accuracy'] = random.random()
        result_json['processing_time'] = random.random()
    else:
        # 因为bayesian optimization是将目标函数变小，所以accuracy变成了1-accuracy
        f_0 = 1-accuracy
        f_1 = (processing_time-min_processing_time) / (max_processing_time-min_processing_time)
        result['objectives'] = np.stack([f_0, f_1], axis=-1)
        result_json['accuracy'] = f_0
        result_json['processing_time'] = f_1
    return result, result_json

def init_knobs():
    # # FPS_CHOICE = ["1", "2", "4", "10"]
    # # fix into 1 framerate, for ablation study
    # FPS_CHOICE = ["1"]
    # RESOLUTION_CHOICE = ["320", "640", "1280"]
    
    # # FPS_RESOLUTION_CHOICE = []
    # # for i in FPS_CHOICE:
    # #     for j in RESOLUTION_CHOICE:
    # #         FPS_RESOLUTION_CHOICE.append(i+'&'+j)
    
    # SIZE_CHOICE = ["yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]
    FPS_CHOICE = ["4"]
    RESOLUTION_CHOICE = ["1280"]
    SIZE_CHOICE = ["yolov5m.pt"]
    
    return FPS_CHOICE, RESOLUTION_CHOICE, SIZE_CHOICE


 
class bayedian_optimization():
    def __init__(self, camera_num, fps_choice, resolotion_choice, size_choice, ref_point=[], cluster_id=0):
        self.ref_point = ref_point
        self.space = sp.Space()
        self.camera_num = camera_num
        self.cluster_id = cluster_id
        self.FPS_CHOICE, self.RESOLUTION_CHOICE, self.SIZE_CHOICE, self.matching_low_bound = fps_choice, resolotion_choice, size_choice, matching_low_bound
        knobs = []
        for i in range(self.camera_num):
            knobs.append(sp.Categorical('fps_camera'+str(i+1), self.FPS_CHOICE))
            knobs.append(sp.Categorical('resolution_camera'+str(i+1), self.RESOLUTION_CHOICE))
            knobs.append(sp.Categorical('model_size_camera'+str(i+1), self.SIZE_CHOICE))
        # q表示采样间隔
        knobs.append(sp.Real('lazy_threshold', matching_low_bound, matching_upper_bound, default_value = 0.7, q = 0.02))
        self.space.add_variables(knobs)
        
    def start_optimize(self):
        t_before_opt = time_synchronized()
        opt = Optimizer(multi_camera_matching, self.space,
                        num_objectives=2,
                        num_constraints=0,
                        max_runs=20,
                        surrogate_type='prf',
                        acq_type='ehvi',
                        acq_optimizer_type='local_random',
                        initial_runs=9,
                        ref_point=self.ref_point,
                        task_id=self.cluster_id)
        history = opt.run()
        t_after_opt = time_synchronized()
        print('优化所用时间：', t_after_opt-t_before_opt)
        print("history记录的打印: ", history)
        # with open(config_file_name, 'a') as file:
        #     json.dump(t_after_opt, file)
        #     json.dump(history, file)
        #     file.write('\n') 

        # plot pareto front
        if history.num_objectives in [2, 3]:
            history.plot_pareto_front()  # support 2 or 3 objectives
            plt.show()

        # plot hypervolume (optimal hypervolume of BraninCurrin is approximated using NSGA-II)
        history.plot_hypervolumes(optimal_hypervolume=59.36011874867746, logy=True)
        plt.show()
        
if __name__ == "__main__":
    # mp.set_start_method(method='spawn',force=True)  # force all the multiprocessing to 'spawn' methods
    FPS_CHOICE, RESOLUTION_CHOICE, SIZE_CHOICE = init_knobs()

    ref_point = [1,1]
    bo = bayedian_optimization(camera_num=camera_num, fps_choice=FPS_CHOICE, resolotion_choice = RESOLUTION_CHOICE, size_choice=SIZE_CHOICE, ref_point=ref_point, cluster_id=0)
    bo.start_optimize() 