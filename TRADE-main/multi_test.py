import os
import sys
import queue
import traceback
import multiprocessing as mp
from multiprocessing.managers import SyncManager

CAM_NUMBER = 80
BUFFER_SIZE = 1

def main():
    proc_list = list()
    manager = SyncManager()
    manager.start()

    traj_qs = [manager.Queue(maxsize=BUFFER_SIZE) for _ in range(CAM_NUMBER)]
    frame_qs = [manager.Queue(maxsize=BUFFER_SIZE) for _ in range(CAM_NUMBER)]
    cnt = 0
    try:
        for i in range(CAM_NUMBER):
            cam_proc = mp.Process(target=worker_cam, args=(i, frame_qs[i], traj_qs[i]))
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

def worker_cam(id:int, frame_q:queue.Queue, traj_q:queue.Queue):
    try:
        while True:
            frame = frame_q.get(block=True, timeout=3)
            traj_q.put(item=[id, frame], block=True, timeout=None)
            if frame == 10:
                break
    except Exception as e:
        print(e.args)
        print(traceback.format_exc())
    finally:
        print(f"CAM WORKER {id} PROCESS EXIT!\n")

if __name__ == '__main__':
    try:
        mp.set_start_method(method='spawn', force=True)
        main()
    finally:
        pass
