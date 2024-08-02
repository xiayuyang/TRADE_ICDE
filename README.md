# [TRADE: Scalable Configuration Tuning for Resource-efficient Cross-camera Video Processing]

![image](/figure/frame.png)

The figure shows the architecture of our cross-camera video processing framework, called TRADE, which includes an innovative cross-camera processing pipeline with multiple knobs and a scalable and transferable configuration tuning module to recommend near-optimal configuration combinations.

Cross-camera Video Processing.
Given a multi-camera video system, \myfw\ adjusts its resource consumption and accuracy by setting knobs. For single-camera video processing, we set multiple knobs for each camera, including framerate and resolution to adjust the amount of video data, and model size to adjust the resource consumption of each processing. For cross-camera object matching, we propose an improved early termination algorithm to reduce the search space of object matching. It leverages spatio-temporal correlations to distinguish the matching probabilities of vehicles and employs a matching threshold as the termination condition of the search. Moreover, the matching threshold is set as a knob in cross-camera object matching.


Configuration Tuning.
This section aims to find near-optimal configuration combinations for the above cross-camera video processing pipeline. Specifically, we first periodically categorize all cameras into $k$ categories, enabling configuration sharing among videos within the same category. Then, we perform configuration tuning in two stages based on multi-objective (i.e., resource consumption and accuracy) Bayesian optimization. The first stage turns multiple single-camera knobs into a composite knob for each video category, and the second stage explores promising configuration combinations of $k$ composite knobs and a cross-camera knob. Finally, we propose a meta-learning strategy, which consists of an initialization strategy and a meta-surrogate model to solve the cold-start issue when tuning a new multi-camera video system. Since the configuration tuning process involves profiling multiple configuration combinations on video data and necessitates comparing produced results with manually annotated labels, online tuning for incoming videos is impractical. Consequently, we train our configuration tuning model offline on a small fraction of video data and build a configuration dictionary, with keys being the various accuracy lower bounds and values being the corresponding most resource-efficient configuration combinations found in the offline tuning phase. In this way, during online video processing, we can directly refer to the configuration dictionary to select a configuration combination based on video categories and a given accuracy lower bound. For some stable multi-camera video systems, such as traffic surveillance systems, we contend that training once a month or less is sufficient to maintain a good tuning performance.

<!-- ## Code Structure
we decribe some core files in the following.
1. detector/yolov5/s_bo.py. 
2. detector/yolov5/bo.py -->



### Datasets and Models
1. CityFlow. A real-world multi-camera video datasets. Due to its data license agreement, we can only download data from its official [website](https://www.aicitychallenge.org/)

2. Synthetic. A synthetic multi-camera video benchmark of paper [Visual Road: A Video Data Management Benchmark](https://dl.acm.org/doi/pdf/10.1145/3299869.3324955). Since the 80-camera video data is too large, we present an example video dataset in [this link](https://drive.google.com/drive/folders/1ueVphZwP3T05uWA3qlRkHzU2FeA1anxf).

3. YOLO and Reid model. The YOLO model is download from [this link](https://github.com/ultralytics/yolov5) and the reid model is download from [this link](https://github.com/Pirazh/SSBVER). and then retrained based on our dataset. 


### Getting Started

```bash
# setup conda and python environment
$ conda create -n env_name python=3.7
$ conda activate env_name

# clone the repo and install the required dependency.
$ git clone  https://github.com/xiayuyang/TRADE_ICDE.git
$ pip install -r ./TRADE-main/requirements.txt

# run the single-camera tuning and cross-camera tuning
$ python ./TRADE-main/detector/yolov5/s_bo.py
$ python ./TRADE-main/detector/yolov5/bo.py
```
