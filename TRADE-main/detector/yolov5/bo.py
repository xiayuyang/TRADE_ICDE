'''
    多视频的tuning
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
from my_mtmc import profile, profile_multi
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
TEST = False
# NOTE
cams_dir = 'datasets/AIC22_Track1_MTMC_Tracking/train/S13'
cams = os.listdir(cams_dir)
camera_num=len(cams)



current_date = datetime.date.today()
# 将日期格式化为字符串，格式为 YYYY-MM-DD
formatted_date = current_date.strftime('%Y-%m-%d')




min_processing_time = 1*60*80
max_processing_time = 10*60*80
matching_low_bound = 0.6
matching_upper_bound = 0.9

CAM_NUMBER = 80
BUFFER_SIZE = 1

# 创建文件名,存每次的config与objective值
config_file_name = f"./tune_result/{formatted_date}.json"
data = dict()
data['min_processing_time'] = min_processing_time
data['max_processing_time'] = max_processing_time
data['matching_low_bound'] = matching_low_bound

data['matching_upper_bound'] = matching_upper_bound
data['log'] = 'config_updated_pp normalize_F1 matching_crossmatch,large configuration space,FPS_CHOICE = 10 RESOLUTION_CHOICE = 5 SIZE_CHOICE = 4'
# 写入 JSON 数据到文件
with open(config_file_name, 'w') as file:
    json.dump(data, file, indent=4)

def read_json(type1, type2, type3, type4):
    single_camera_1 = f"./single_camera_config/type1.txt"
    single_camera_2 = f"./single_camera_config/type2.txt"
    single_camera_3 = f"./single_camera_config/type3.txt"
    single_camera_4 = f"./single_camera_config/type4.txt"
    weights = []
    imgsz = []
    cam_ratio = []
    line_list = []
    with open(single_camera_1, 'r') as file1:
        lines1 = file1.readlines()
        line_list.append(lines1)
    with open(single_camera_2, 'r') as file2:
        lines2 = file2.readlines()
        line_list.append(lines2)
    with open(single_camera_3, 'r') as file3:
        lines3 = file3.readlines()
        line_list.append(lines3)
    with open(single_camera_4, 'r') as file4:
        lines4 = file4.readlines()
        line_list.append(lines4)
    for line in line_list:
        parts = line.strip().split()
        cam_ratio.append(int(parts[0]))
        imgsz.append(int(parts[1]))
        weights.append(str(parts(2)))
    return weights, imgsz, cam_ratio

# 可以考虑把processing_time与accuracy归一化一下, 
# 并且把accuracy取反一下，因为openbox默认是最小化目标函数
def multi_camera_matching(config: sp.Configuration):
    params = config.get_dictionary().copy()
    type1 = params['type1']
    type2 = params['type2']
    type3 = params['type3']
    type4 = params['type4']
    weights, imgsz, cams_ratio = read_json(type1, type2, type3, type4)
    lazy_threshold = params['lazy_threshold']
    with open(config_file_name, 'a') as file:
        json.dump(params, file)
        file.write('\n') 
    if TEST:
        result, result_json = normalize_objective(test=True)
    else:
        proc_list = list()
        manager = SyncManager()
        manager.start()
        traj_qs = [manager.Queue(maxsize=BUFFER_SIZE) for _ in range(CAM_NUMBER)]
        frame_qs = [manager.Queue(maxsize=BUFFER_SIZE) for _ in range(CAM_NUMBER)]
        cnt = 0
        try:
            for i in range(CAM_NUMBER):
                # i%4是GPU_ID, i%16是进程ID
                cam_proc = mp.Process(target=profile_multi, args=(i%4, i%16, frame_qs[i], traj_qs[i], cams_dir, weights, imgsz, cams_ratio, lazy_threshold))
                cam_proc.start()
                proc_list.append(cam_proc)

            while True:
                for i in range(CAM_NUMBER):
                    frame_qs[i].put(item=cnt, block=True, timeout=None)
                for i in range(CAM_NUMBER):
                    traj = traj_qs[i].get(block=True, timeout=None)
                    print(traj)
                if cnt == 10:
                    break
                cnt += 1

        except Exception as e:
            print(e.args)
            print(traceback.format_exc())
        finally:
            [p.join() for p in proc_list]
            manager.shutdown()
            print("DONE, EXIT MAIN PROCESS!\n")
        accuracy, processing_time, section_json = profile(
            cams_dir_= cams_dir, weights_= weights, imgsz_=imgsz, cams_ratio_=cams_ratio, lazy_threshold_=lazy_threshold)
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
    # it only represents categorizes.
    type1 = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    type2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    type3 = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    type4 = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    return type1, type2, type3, type4


 
class bayedian_optimization():
    def __init__(self, camera_num, choise1, choise2, choise3, choise4, ref_point=[], cluster_id=0):
        self.ref_point = ref_point
        self.space = sp.Space()
        self.camera_num = camera_num
        self.cluster_id = cluster_id
        knobs = []
        knobs.append(sp.Categorical('type1', choise1))
        knobs.append(sp.Categorical('type2', choise2))
        knobs.append(sp.Categorical('type3', choise3))
        knobs.append(sp.Categorical('type4', choise4))
        # q表示采样间隔
        knobs.append(sp.Real('lazy_threshold', matching_low_bound, matching_upper_bound, default_value = 0.7, q = 0.02))
        self.space.add_variables(knobs)
        
    def start_optimize(self):
        t_before_opt = time_synchronized()
        opt = Optimizer(multi_camera_matching, self.space,
                        num_objectives=2,
                        num_constraints=0,
                        max_runs=200,
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
    type1, type2, type3, type4 = init_knobs()

    ref_point = [1,1]
    mp.set_start_method(method='spawn', force=True)
    bo = bayedian_optimization(camera_num=camera_num, choise1 = type1, choise2 = type2, choise3 = type3, choise4 =type4,  ref_point=ref_point, cluster_id=0)
    bo.start_optimize() 