'''
Rui Zhu

python dump_openrooms_multi.py --gpu_total 8

, to dump all scenes in parallel; does not necessarily bring 8x speedup, but hopefully no worries of GPU issues with multiprocessing + mitsuba/torch

Simple multi-thread testing: python test_scripts/test_cuda_multi_3.py
'''

# from tqdm import tqdm
import numpy as np
np.set_printoptions(suppress=True)
import os, sys
import torch
# import torch.multiprocessing.spawn as spawn
import torch.multiprocessing as mp
import time
import subprocess

PATH_HOME = '/home/ruizhu/Documents/Projects/rui-indoorinv-data'
sys.path.insert(0, PATH_HOME)

from lib.utils_misc import run_cmd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_total', type=int, default=8, help='total num of gpus available')
# parser.add_argument('--workers_total', type=int, default=-1, help='total num of workers; must be dividable by gpu_total, i.e. workers_total/gpu_total jobs per GPU')
opt = parser.parse_args()

def run_one_gpu(i, opt, split, result_queue):
    torch.cuda.is_available()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
    print("pid={} count={}".format(i, torch.cuda.device_count()))
    
    # process_result_list = []
    cmd = 'python tools/Mask3D/dump_openrooms_func.py --gpu_id {} --gpu_total {} --split {}'.format(i, opt.gpu_total, split)
    _results = run_cmd(cmd)
    print(_results)
        
    # return process_result_list
    result_queue.put((i, _results))

if __name__ == '__main__':
    
    # if opt.workers_total == -1:
    #     opt.workers_total = opt.gpu_total
    
    # for split in ['train', 'val']:
    for split in ['val']:
        tic = time.time()
        
        result_queue = mp.Queue()
        for rank in range(opt.gpu_total):
            mp.Process(target=run_one_gpu, args=(rank, opt, split, result_queue)).start()
        
        for _ in range(opt.gpu_total):
            temp_result = result_queue.get()
            print(_, temp_result)
            
        print('==== ...DONE. Took %.2f seconds'%(time.time() - tic))