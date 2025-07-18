##########################################################################################
# Machine Environment Config
import random
import time
import numpy as np
import torch

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


import logging
from utils.utils import create_logger, copy_all_src

from CARPTester import CARPTester as Tester


##########################################################################################
# parameters
env_params = {
    'vertex_size': 30,
    'edge_size': 50,
    'pomo_size': 50,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',  # softmax
}


tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/AEGT_E50.pt',  # directory path of pre-trained model and log files saved.
    },

    'test_data_load': {
        'enable': True,
        'type': 'SyntheticData',  # SyntheticData, RealWorldData, BenchmarkData
        'filename': './TestData/SyntheticData/E50'
    },
    'test_episodes': 1000,
    'test_batch_size': 1000,
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    _print_config()


    seed = 1234  # 可以是任何你选择的整数种子
    # 设置 Python 随机种子
    random.seed(seed)
    # 设置 NumPy 随机种子
    np.random.seed(seed)
    # 设置 PyTorch 随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



    # 创建 Tester 实例
    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)


    tester.run()



def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
