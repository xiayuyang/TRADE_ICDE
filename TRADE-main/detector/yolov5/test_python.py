# # out time
# import itertools
# import numpy as np

# # 统计每个gt文件中的轨迹id及出现的时间
# cam_names = ['c001', 'c002', 'c003', 'c004']
# # timestamp
# timestamp_file = 'datasets/AIC22_Track1_MTMC_Tracking/cam_timestamp/S12.txt'
# timestamp_dict = {}
# with open(timestamp_file, 'r') as time_file:
#     for line in time_file:
#         cam_name, stamp = line.strip().split(' ')
#         timestamp_dict[cam_name] = int(float(stamp) * 10)
# data = {} 
# for cam in cam_names:
#     gt_file = f'datasets/AIC22_Track1_MTMC_Tracking/train/S12/{cam}/gt/gt.txt'
#     with open(gt_file, 'r') as file:
#         for line in file:
#             # 以检测到的最后一帧为时间起点
#             last_frame, tid = list(line[:-12].strip().split(','))[:2]
#             if (cam not in data.keys()):
#                 data[cam] = {}
#             data[cam][tid] = int(last_frame) + timestamp_dict[cam]

# timestamp_differences_dict = {}

# # 遍历所有摄像头组合
# for (camera1, trajectories1), (camera2, trajectories2) in itertools.combinations(data.items(), 2):
#     differences = []
    
#     # 遍历第一个摄像头的轨迹
#     for tid, timestamp1 in trajectories1.items():
#         # 如果轨迹在第二个摄像头也存在，则计算timestamp差值
#         if tid in trajectories2:
#             timestamp2 = trajectories2[tid]
#             difference = abs(timestamp2 - timestamp1)
#             differences.append(difference / 10.0)

#     key = (camera1, camera2)
#     timestamp_differences_dict[key] = differences

# # plot the distribution of differences for each camera pair baased on timestamp_differences_dict
# import matplotlib.pyplot as plt
# import seaborn as sns
# for (camera1, camera2), differences in timestamp_differences_dict.items():
#     plt.figure()
#     sns.histplot(differences, kde=True, alpha=0)
#     plt.title(f"Timestamp differences between {camera1} and {camera2}")
#     plt.xlabel("Time difference (s)")
#     plt.ylabel("Frequency")
#     plt.show()
#     # save this figure
#     plt.savefig(f"Timestamp differences between {camera1} and {camera2}.png")



# # for cameras, differences in timestamp_differences_dict.items():
# #     print(f"Timestamp differences between {cameras[0]} and {cameras[1]}:")
# #     print(np.mean(differences))


# time diff = in time - out time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# 统计每个gt文件中的轨迹id及出现的时间

vdo_rate = 10
cams_path = 'datasets/AIC22_Track1_MTMC_Tracking/train/S03'
timestamp_file = 'datasets/AIC22_Track1_MTMC_Tracking/cam_timestamp/S03.txt'

cam_names = os.listdir(cams_path)
timestamp_dict = {}
with open(timestamp_file, 'r') as time_file:
    for line in time_file:
        cam_name, stamp = line.strip().split(' ')
        timestamp_dict[cam_name] = int(float(stamp) * vdo_rate)
data = {}
for cam in cam_names:
    gt_file = f'{cams_path}/{cam}/gt/gt.txt'
    with open(gt_file, 'r') as file:
        for line in file:
            # 以检测到的最后一帧为时间起点
            last_frame, tid = list(line[:-12].strip().split(','))[:2]
            if (cam not in data.keys()):
                data[cam] = {}
            if tid not in data[cam]:
                # 记录in_time
                data[cam][tid] = {}
                data[cam][tid][0] = int(last_frame) - timestamp_dict[cam]
            # 记录out_time
            data[cam][tid][1] = int(last_frame) - timestamp_dict[cam]

timestamp_differences_dict = {}

# 遍历所有摄像头组合
for (camera1, trajectories1), (camera2, trajectories2) in itertools.combinations(data.items(), 2):
    differences = []

    # 遍历第一个摄像头的轨迹
    for tid, time1 in trajectories1.items():
        # 如果轨迹在第二个摄像头也存在，则计算timestamp差值
        if tid in trajectories2:
            time2 = trajectories2[tid]
            if time1[0] > time2[0]:
                difference = time1[0] - time2[1]
            else:
                difference = time2[0] - time1[1]
            differences.append(difference / vdo_rate)

    key = (camera1, camera2)
    timestamp_differences_dict[key] = differences

for cameras, differences in timestamp_differences_dict.items():
    if cameras[0] == 'c033' and cameras[1] == 'c036':
        plt.histplot(differences, bins=30, alpha=0.5, color='b')
        plt.title('Histogram')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.savefig('1.png')
    if cameras[0] == 'c033' and cameras[1] == 'c040':
        plt.histplot(differences, bins=30, alpha=0.5, color='b')
        plt.title('Histogram')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.savefig('2.png')
    print(f"Timestamp differences between {cameras[0]} and {cameras[1]}:")
    print(np.mean(differences))