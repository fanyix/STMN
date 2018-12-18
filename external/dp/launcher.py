import subprocess
import os
import numpy as np
import time

# number of procs
proc_N = 12
gpu_id = np.array([0,1,2,3,4,5,6,7,0,1,2,3])

assert proc_N == len(gpu_id)

# launch processes
for proc in range(0, proc_N):
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = '%g' % gpu_id[proc]
    my_env["CUDNN_PATH"] = '/usr/local/cuda/lib64/libcudnn.so.5'
    #my_env["THC_CACHING_ALLOCATOR"] = '0' # '0' for memory-saving mode
    subprocess.Popen(['nohup', 'matlab', '-nodisplay', '-nosplash', '-r', "run('run_dp.m');exit;", '>', 'logs/out_%d' % proc, '2>&1', '&'], env=my_env)
    print '%g/%g process launched' % (proc, proc_N)
    time.sleep(10)



