import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--metric', action='store_true', help='get the metric of major kernels')
args = parser.parse_args()


metrics = ['flop_count_sp', 'flop_count_sp_special']#'all']
kernels = ['fusedL2Normv2', 'fusedL2Normb', 'bn_fw', 'bn_bw']


for i in range(9):
    if args.metric:
        cmd = "nvprof -o norm%d_m.nvvp -f --profile-from-start off" % i
        for k in kernels:
            cmd += ' --kernels %s' % k
            for m in metrics:
                cmd += ' --metrics %s' % m
    else:
        cmd = "nvprof -o norm%d.nvvp -f --profile-from-start off" % i
    
    cmd += ' python -m moment_norm.benchmark --bm %d' % i
    
    print(cmd)
    os.system(cmd)
    